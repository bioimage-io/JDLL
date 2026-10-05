"""Bounded, mask-first planning of distinct StarDist validation patches."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from stardist_data import fingerprint, sample_start, statistics


def build_2d_plan(labels, patch, requested, foreground, rng, check):
    """Expand by usable image area, preserving source coverage and bounded work."""
    if any(any(n < p for n, p in zip(y.shape, patch)) for y in labels):
        raise ValueError('StarDist validation sources must be at least as large as train_patch_size.')
    capacity = np.array([math.prod(n // p for n, p in zip(y.shape, patch)) for y in labels], dtype=np.int64)
    count = min(max(requested, len(labels)), int(capacity.sum()))
    quotas = np.ones(len(labels), dtype=np.int64)
    remaining = count - len(labels)
    if remaining:
        shares = (capacity - 1).astype(np.float64) * remaining / (capacity - 1).sum()
        extra = np.floor(shares).astype(np.int64)
        quotas += extra
        order = np.argsort(-(shares - extra), kind='stable')
        quotas[order[:remaining - int(extra.sum())]] += 1
    indices = np.repeat(np.arange(len(labels)), quotas)
    rng.shuffle(indices)
    plan, seen = [], set()
    for index in indices:
        i = int(index)
        # Keep the existing foreground policy, but do not inflate validation with
        # duplicate coordinates when foreground-constrained space is limited.
        for _ in range(16):
            check()
            start = sample_start(labels[i], patch, rng, rng.rand() < foreground)
            key = (i, start)
            if key not in seen:
                seen.add(key)
                plan.append(key)
                break
    return plan


def build_plan(images, labels, patch, grid, requested, options, rng, check, path):
    fraction = float(options.get('foreground_fraction', 0.33))
    occupancy = float(options.get('minimum_foreground', 0.01))
    coverage = float(options.get('minimum_source_fraction', 0.5))
    overlap = float(options.get('max_sampling_overlap', 0.1))
    attempts = int(options.get('candidate_attempts', 16))
    if not (0 <= fraction <= 1 and 0 < occupancy <= 1 and 0 <= coverage <= 1
            and 0 <= overlap <= 0.1 and attempts >= 1):
        raise ValueError('Invalid validation sampling settings.')
    if any(any(n < p for n, p in zip(y.shape, patch)) for y in labels):
        raise ValueError('StarDist validation sources must be at least as large as train_patch_size.')
    policy = {k: options[k] for k in ('seed', 'foreground_fraction', 'minimum_foreground',
              'minimum_source_fraction', 'max_sampling_overlap', 'candidate_attempts', 'resolved_context') if k in options}
    identity = dict(sources=fingerprint(images, labels), patch=list(patch), grid=list(grid),
                    requested=requested, policy=policy)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    previous = Path(options.get('plan_path', path))
    if previous.is_file():
        saved = json.loads(previous.read_text())
        if saved.get('signature') == signature:
            return [(int(i), tuple(s)) for i, s in saved['patches']], dict(saved['summary'], reused=True), signature

    foreground_sources = {i for i, y in enumerate(labels) if len(statistics(y)['foreground'])}
    measured = {}

    def candidates(allowed_overlap):
        axes, counts = [], []
        for label in labels:
            positions = []
            for n, p, g in zip(label.shape, patch, grid):
                step = min(p, max(g, int(math.ceil(p * (1 - allowed_overlap) / g)) * g))
                count = 1 + (n - p) // step
                slack = n - p - (count - 1) * step
                offset = int(rng.randint(slack // g + 1)) * g
                positions.append((offset, step, count))
            axes.append(positions)
            counts.append(math.prod(a[2] for a in positions))
        # Allocate search effort across sources without constructing a giant grid.
        limit = max(requested * attempts, len(labels))
        indices = list(range(len(labels)))
        rng.shuffle(indices)
        result = []
        for i in indices:
            check()
            take = min(counts[i], max(1, int(math.ceil(limit * counts[i] / sum(counts)))))
            if take == counts[i]:
                chosen = range(counts[i])
            else:
                chosen = set()
                while len(chosen) < take:
                    chosen.add(int(rng.randint(counts[i])))
            shape = tuple(a[2] for a in axes[i])
            for flat in chosen:
                check()
                cell = np.unravel_index(flat, shape)
                start = tuple(int(a[0] + c * a[1]) for a, c in zip(axes[i], cell))
                key = (i, start)
                if key not in measured:
                    mask = labels[i][tuple(slice(s, s + p) for s, p in zip(start, patch))]
                    valid = np.count_nonzero(mask >= 0)
                    measured[key] = np.count_nonzero(mask > 0) / valid if valid else None
                if measured[key] is not None:
                    result.append((i, start, measured[key]))
        rng.shuffle(result)
        return result

    pool = candidates(0)
    used_overlap = 0.0
    if len(pool) < requested and overlap:
        alternative = candidates(overlap)
        if len(alternative) > len(pool):
            pool, used_overlap = alternative, overlap
    count = min(requested, len(pool))
    if count == 0:
        raise ValueError('No validation patch with valid targets fits the configured patch size.')
    limited = count < requested
    quota = 0 if limited else int(math.ceil(fraction * count))
    desired_sources = int(math.ceil(coverage * len(foreground_sources))) if quota else 0
    required_sources = min(quota, desired_sources)
    qualified = [p for p in pool if p[2] >= occupancy]
    resolved_occupancy = occupancy
    if quota and (len(qualified) < quota or len({p[0] for p in qualified}) < required_sources):
        positive = sorted((p for p in pool if p[2] > 0), key=lambda p: p[2], reverse=True)
        if positive:
            thresholds = [positive[min(quota, len(positive)) - 1][2]]
            source_best = sorted((max(p[2] for p in positive if p[0] == i)
                                  for i in {p[0] for p in positive}), reverse=True)
            if required_sources:
                thresholds.append(source_best[min(required_sources, len(source_best)) - 1])
            resolved_occupancy = min(occupancy, *thresholds)
            qualified = [p for p in positive if p[2] >= resolved_occupancy]
    chosen, chosen_keys = [], set()

    def add(candidate):
        key = candidate[:2]
        if key not in chosen_keys and len(chosen) < count:
            chosen.append(candidate)
            chosen_keys.add(key)

    # First spread the forced quota across foreground-containing sources.
    source_order = list(dict.fromkeys(p[0] for p in qualified))
    rng.shuffle(source_order)
    for i in source_order[:required_sources]:
        add(next(p for p in qualified if p[0] == i))
    for p in qualified:
        if len(chosen) >= quota:
            break
        add(p)
    forced = len(chosen)
    forced_sources = len({p[0] for p in chosen})
    for p in pool:
        add(p)
        if len(chosen) == count:
            break
    reasons = []
    if limited:
        reasons.append('limited distinct space found by bounded grid search; foreground quota disabled')
    if quota and resolved_occupancy < occupancy:
        reasons.append('foreground occupancy relaxed to the highest threshold found that supports the quota/coverage')
    if forced < quota or forced_sources < desired_sources:
        reasons.append('foreground quota or source coverage unattainable in bounded search')
    summary = dict(requested_samples=requested, achieved_samples=count, requested_foreground_samples=quota,
                   achieved_foreground_samples=forced, minimum_foreground=resolved_occupancy,
                   requested_foreground_sources=desired_sources, achieved_foreground_sources=forced_sources,
                   eligible_foreground_sources=len(foreground_sources), sources_covered=len({p[0] for p in chosen}),
                   eligible_sources=len(labels), maximum_overlap=used_overlap, small_capacity=limited,
                   fallback_reasons=reasons, search='bounded_aligned_grid', candidate_masks_checked=len(measured), reused=False)
    return [(i, start) for i, start, _ in chosen], summary, signature
