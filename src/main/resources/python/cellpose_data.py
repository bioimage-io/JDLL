"""Prepare native Cellpose 2D training fields without writing to the source dataset."""
from dataclasses import replace
from pathlib import Path
import random

import numpy as np
import tifffile
import torch
from cellpose import dynamics, transforms
from jdll_unet.crop_reading import CropArray
from jdll_unet.geometry import (DomainReader, assert_disjoint, inspect_pair, load_domain_mask,
                                source_identity, split_sources)
from jdll_unet.image_reading import image_reading_session
from jdll_unet.io import DatasetSplits, ImageMaskPair, discover_dataset
from skimage.measure import label
from jdll_training_continuation import atomic_json, read_json, validation_signature


def single_holdout(pair, reader):
    """Choose a spatial split near 80/20 with foreground on both sides and no cut object."""
    mask = load_domain_mask(pair, reader=reader, raw=True)
    for axis in sorted(range(mask.ndim), key=lambda i: -mask.shape[i]):
        view = np.moveaxis(mask, axis, 0)
        if view.shape[0] < 2:
            continue
        # Adjacent equal nonzero IDs at the seam indicate a sectioned cell.
        crosses = np.any((view[:-1] == view[1:]) & (view[:-1] != 0),
                         axis=tuple(range(1, view.ndim)))
        positive = np.cumsum(np.count_nonzero(view, axis=tuple(range(1, view.ndim))))
        for cut in sorted(np.flatnonzero(~crosses) + 1, key=lambda c: abs(c / len(view) - .8)):
            minimum = 1 if mask.ndim == 3 and axis == 0 else 2
            if min(cut, len(view) - cut) < minimum or not 0 < positive[cut - 1] < positive[-1]:
                continue
            left = list(pair.region or [(0, n) for n in mask.shape])
            right = left.copy()
            start, end = left[axis]
            cut = start + int(cut)
            left[axis], right[axis] = (start, cut), (cut, end)
            return [replace(pair, region=tuple(left))], [replace(pair, region=tuple(right))]
    raise ValueError("No spatial holdout with separate, complete cells on both sides. "
                     "Provide a separate val/images and val/masks pair.")


def source_splits(root, reader, notify, check, sources, run=None):
    if sources is None:
        splits = discover_dataset(root)
    else:
        def pairs(name):
            return [ImageMaskPair(Path(image), Path(mask), Path(image).stem) for image, mask in sources[name]]
        splits = DatasetSplits(pairs("train"), pairs("val"), bool(sources["val"]))
    if (root / "val").exists() and not splits.explicit_val:
        if any(p.is_file() and p.suffix.lower() in {".png", ".tif", ".tiff", ".npy", ".jpeg", ".jpg", ".bmp"}
               for p in (root / "val").rglob("*")):
            raise ValueError("Incomplete validation set: supply both val/images and val/masks.")
        notify("Validation is empty. Creating a holdout from the current training data.")
    def inspect(pairs):
        result = []
        seen_images, seen_masks = set(), set()
        for pair in pairs:
            check()
            notify("Checking " + pair.image.name)
            verified, _ = inspect_pair(pair, reader=reader)
            mid = source_identity(pair.mask)
            if verified.source_id in seen_images or mid in seen_masks:
                raise ValueError("Duplicate image or mask in dataset: " + str(pair.image))
            seen_images.add(verified.source_id)
            seen_masks.add(mid)
            result.append(verified)
        return result
    train = inspect(splits.train)
    previous = read_json(run.parent / "split.json") if run and run.parent else {}
    if run and run.parent and not splits.explicit_val and not previous:
        notify("No saved automatic split. Rebuilding validation from the current dataset.")
    def identity(pair):
        return (pair.image.relative_to(root).as_posix(), pair.mask.relative_to(root).as_posix())
    def positive(pair):
        if pair.region:
            return bool(load_domain_mask(pair, reader=reader, raw=True).any())
        return any(pair.plane_positive_counts)
    if splits.explicit_val:
        val = inspect(splits.val)
    elif previous and (len(train) > 1 or previous.get("spatial")):
        known = {tuple(key): group for group in ("train", "val") for key in previous[group]}
        fresh = [pair for pair in train if identity(pair) not in known]
        random.Random(42).shuffle(fresh)
        val = [pair for pair in train if known.get(identity(pair)) == "val"]
        train = [pair for pair in train if known.get(identity(pair)) == "train"]
        count = round(.2 * len(fresh))
        if not val:
            notify("The previous validation sources are unavailable. Rebuilding the validation split.")
            target = max(1, round(.2 * (len(train) + len(fresh))))
            count = min(len(fresh), target)
            needed = min(target - count, max(0, len(train) + len(fresh) - count - 1))
            if needed:
                random.Random(42).shuffle(train)
                val.extend(train[:needed])
                del train[:needed]
                notify("Included previously trained sources in the rebuilt holdout; validation may be optimistic.")
        if not train:
            count = min(count, max(0, len(fresh) - 1))
        val.extend(fresh[:count])
        train.extend(fresh[count:])
        if not val and len(train) > 1:
            train, val = split_sources(train, .2, 42)
            notify("Rebuilt validation from previously trained sources; its scores may be optimistic.")
        if previous.get("spatial"):
            by_source = {identity(pair): pair for pair in train + val}
            for name in {tuple(r["source"]) for r in previous["spatial"]}:
                pair = by_source.get(name)
                if pair is None:
                    continue
                saved = [entry for entry in previous["spatial"] if tuple(entry["source"]) == name]
                if any(len(r["region"]) != len(pair.spatial_shape) or any(
                       not 0 <= a < b <= n for (a,b),n in zip(r["region"],pair.spatial_shape)) for r in saved):
                    continue
                train = [p for p in train if identity(p) != name]
                val = [p for p in val if identity(p) != name]
                for entry in saved:
                    region = replace(pair, region=tuple(tuple(v) for v in entry["region"]))
                    (train if entry["split"] == "train" else val).append(region)
        if not train and len(val) > 1:
            train.append(val.pop())
    else:
        positives = [pair for pair in train if any(pair.plane_positive_counts)]
        empty = [pair for pair in train if not any(pair.plane_positive_counts)]
        if not positives:
            raise ValueError("No annotated cells in training data")
        if len(positives) == 1:
            train, val = single_holdout(positives[0], reader)
        else:
            train, val = split_sources(positives, .2, 42)
        # Pure background can expose false positives during validation.
        val.extend(empty)
    if not any(map(positive, val)):
        annotated = sorted((p for p in train if positive(p)), key=lambda p: bool(p.region))
        if len(annotated) >= 2:
            candidate = next((p for p in annotated if not p.region and list(identity(p)) not in previous.get("train", [])), annotated[0])
            train.remove(candidate)
            val.append(candidate)
            notify(f"Validation contained no annotated cells. Moved {candidate.image.name} from training into validation.")
        elif annotated:
            left, right = single_holdout(annotated[0], reader)
            train.remove(annotated[0])
            train.extend(left)
            val.extend(right)
            notify("Only one annotated source remains. Reserved a disjoint region for validation.")
    elif not any(map(positive, train)):
        annotated = [p for p in val if positive(p)]
        if len(annotated) >= 2:
            candidate = annotated[-1]
            val.remove(candidate)
            train.append(candidate)
            notify(f"Moved {candidate.image.name} into training so both splits contain annotations.")
        elif annotated:
            left, right = single_holdout(annotated[0], reader)
            val.remove(annotated[0])
            train.extend(left)
            val.extend(right)
            notify("Reserved disjoint training and validation regions in the only annotated source.")
    if run:
        if not splits.explicit_val:
            atomic_json(run.directory / "split.json", dict(
                train=[identity(p) for p in train], val=[identity(p) for p in val],
                spatial=[dict(source=identity(p), region=p.region, split=s)
                         for s, group in (("train", train), ("val", val)) for p in group if p.region]))
        run.info["validation"] = validation_signature(root, [(p.image, p.mask, p.region) for p in val])
        run.flush()
    assert_disjoint(train, val)
    return train, val


def pack_channels(image, channels):
    """Match the C,Y,X channel selection in CellposeChannels.pack."""
    primary, secondary = channels
    if min(channels) < 0 or max(channels) > image.shape[0]:
        raise ValueError(f"Channels {channels} unavailable in a {image.shape[0]}-channel field")
    packed = np.zeros((2, *image.shape[1:]), dtype=np.float32)
    packed[0] = image[primary - 1] if primary else np.mean(image, axis=0, dtype=np.float32)
    if secondary:
        packed[1] = image[secondary - 1]
    return packed


def prepare_dataset(root, work, channels, notify, check, sources=None, check_space=None, run=None):
    # Native train_seg(load_files=False) loads only the fields used by the current batch.
    # Normalize each full field once; native training still owns diameter scaling and augmentation.
    records = {"train": [], "val": []}
    with image_reading_session(lambda kind, **e: notify(e["message"]) if e.get("message") else None) as session:
        reader = DomainReader(session=session)
        train, val = source_splits(Path(root), reader, notify, check, sources, run)
        if check_space is not None:
            check_space(train, val)
        for split, pairs in (("train", train), ("val", val)):
            skipped = 0
            for pair in pairs:
                check()
                volume = len(pair.domain_shape) == 3
                for z in range(pair.domain_shape[0] if volume else 1):
                    check()
                    center = z if volume else None
                    plane = CropArray(pair, reader, mask=True, center_z=center)[...]
                    if split == "train" and not plane.any():
                        skipped += 1
                        continue
                    labels = label(plane, connectivity=1, background=0).astype(np.int32)
                    count = int(labels.max())
                    if count >= 2**24:
                        raise ValueError(f"Too many instances for native float32 flow labels: {pair.mask}")
                    image = CropArray(pair, reader, center_z=center, statistics={"type": "none"})[...]
                    packed = pack_channels(image, channels)
                    if min(plane.shape) < 2:
                        raise ValueError(f"Cellpose requires a 2D field at least two pixels wide and high: {pair.image}")
                    packed = transforms.normalize_img(packed, axis=0)
                    if not np.all(np.isfinite(packed)):
                        raise ValueError(f"Non-finite normalized intensities in {pair.image} (plane {z})")
                    index = len(records[split])
                    stem = work / f"{split}-{index:06d}"
                    image_path, flow_path = str(stem) + ".tif", str(stem) + "-flows.tif"
                    tifffile.imwrite(image_path, packed, photometric="minisblack")
                    # Full instance geometry is still available here, before Cellpose samples crops.
                    flows = dynamics.labels_to_flows([labels], device=torch.device("cpu"))[0]
                    tifffile.imwrite(flow_path, flows, photometric="minisblack")
                    records[split].append(dict(image=image_path, flow=flow_path, source=str(pair.image),
                                               mask=str(pair.mask), plane=z, region=pair.region, instances=count))
                notify(f"Prepared {pair.image.name} ({split})")
            if skipped:
                notify(f"Skipped {skipped} empty training planes")
            if not any(r["instances"] for r in records[split]):
                raise ValueError(f"No annotated cells in the {split} split")
    return records
