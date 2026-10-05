"""Disk-backed StarDist sources; only sampled crops become normalized float arrays."""

import atexit
import hashlib
import math
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np


def available_memory():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def blocks(shape, maximum=262144):
    """Spatial slabs with bounded workspace, including wide single-plane images."""
    import itertools
    chunk = list(shape)
    for axis in range(len(shape)):
        chunk[axis] = max(1, min(chunk[axis], maximum // max(1, math.prod(chunk[axis + 1:]))))
    for start in itertools.product(*(range(0, n, c) for n, c in zip(shape, chunk))):
        yield tuple(slice(s, min(n, s + c)) for s, n, c in zip(start, shape, chunk))


def mask_statistics(array, check=lambda: None):
    """One blockwise pass: counts, boxes, and a bounded foreground reservoir."""
    objects = {}
    points, priorities = np.empty(0, np.int64), np.empty(0)
    rng = np.random.RandomState(42)
    for region in blocks(array.shape):
        check()
        tile = np.asarray(array[region])
        if not np.all(np.isfinite(tile)) or np.any(tile != np.floor(tile)) or np.any(tile < -1):
            raise ValueError('StarDist masks must contain integer instance IDs (or -1 for ignored targets).')
        if tile.max(initial=0) > np.iinfo(np.int32).max:
            raise ValueError('StarDist instance IDs must fit int32; relabel the mask before training.')
        coordinates = np.column_stack(np.nonzero(tile > 0))
        if not len(coordinates):
            continue
        origin = np.array([s.start for s in region])
        values = tile[tuple(coordinates.T)].astype(np.int64)
        ids, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        low = np.full((len(ids), tile.ndim), np.iinfo(np.int64).max, dtype=np.int64)
        high = np.zeros_like(low)
        for axis in range(tile.ndim):
            np.minimum.at(low[:, axis], inverse, coordinates[:, axis] + origin[axis])
            np.maximum.at(high[:, axis], inverse, coordinates[:, axis] + origin[axis] + 1)
        for label, count, lo, hi in zip(ids, counts, low, high):
            previous = objects.get(int(label))
            objects[int(label)] = (int(count) + (previous[0] if previous else 0),
                                  np.minimum(lo, previous[1]) if previous else lo,
                                  np.maximum(hi, previous[2]) if previous else hi)
        flat = np.ravel_multi_index((coordinates + origin).T, array.shape)
        points = np.concatenate((points, flat))
        priorities = np.concatenate((priorities, rng.random_sample(len(flat))))
        if len(points) > 8192:
            keep = np.argpartition(priorities, 8191)[:8192]
            points, priorities = points[keep], priorities[keep]
    return {'objects': objects, 'foreground': points}


class DatasetStore:
    def __init__(self, read, canonical, dimensions, channels, options, update, cancelled):
        self.read, self.canonical = read, canonical
        self.dimensions, self.channels = dimensions, channels
        self.options, self.update, self.cancelled = options, update, cancelled
        budget = options.get('cache_mb', 128)
        self.budget = int(float(budget) * 1024**2)
        if self.budget < 0:
            raise ValueError('data_loading.cache_mb must be nonnegative.')
        self.directory = Path(tempfile.mkdtemp(prefix='jdll-stardist-data-', dir=options.get('cache_dir')))
        self.cache, self.bytes = OrderedDict(), 0
        self.analysis_cache, self.analysis_bytes = OrderedDict(), 0
        self.analysis_budget = int(float(options.get('statistics_cache_mb', 32)) * 1024**2)
        if self.analysis_budget < 0:
            raise ValueError('data_loading.statistics_cache_mb must be nonnegative.')
        self.lock = threading.RLock()
        self.sources = {}
        self.mapped = {}
        atexit.register(self.close)

    def check(self):
        if self.cancelled():
            raise InterruptedError('StarDist dataset preparation cancelled.')

    def decode_path(self, path, size=0):
        """tifffile can decode compressed strips/tiles directly to this file-backed output."""
        key = hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()
        self.check_disk(size)
        return str(self.directory / (key + '.decoded'))

    def check_disk(self, size):
        if shutil.disk_usage(self.directory).free < size + 256 * 1024**2:
            raise OSError('Insufficient disk space for the StarDist decoded-data cache at ' + str(self.directory))

    def remember_mapping(self, key, array):
        base = array
        while base is not None and not isinstance(base, np.memmap):
            base = getattr(base, 'base', None)
        if base is None:
            return False
        offset = base.offset + array.ctypes.data - base.ctypes.data
        self.mapped[key] = (str(base.filename), offset, array.dtype, array.shape, array.strides)
        return True

    def open_mapping(self, key):
        filename, offset, dtype, shape, strides = self.mapped[key]
        span = sum((n - 1) * s for n, s in zip(shape, strides)) + dtype.itemsize
        backing = np.memmap(filename, dtype=dtype, mode='r', offset=offset, shape=(span // dtype.itemsize,))
        return np.ndarray(shape, dtype=dtype, buffer=backing, strides=strides)

    def raster_guard(self, path, image):
        available = available_memory()
        estimate = image.width * image.height * max(1, len(image.getbands())) * 8
        if available is not None and estimate > available * 0.4:
            raise MemoryError('Decoding %s requires too much memory. Export it as a tiled or uncompressed TIFF.' % path)

    def raw(self, source, mask):
        key = (str(source.mask if mask else source.image), mask)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            if key in self.mapped:
                return self.open_mapping(key)
            destination = self.directory / (hashlib.sha256(repr(key).encode()).hexdigest() + '.npy')
            if destination.exists():
                return np.load(destination, mmap_mode='r')
            self.check()
            path = source.mask if mask else source.image
            array, axes = self.read(path, is_mask=mask)
            array = self.canonical(array, axes, mask)
            if not mask:
                array = array[..., :self.channels]
            spatial = array.shape if mask else array.shape[:-1]
            if source.shape is not None and tuple(spatial) != source.shape:
                raise ValueError('Image/mask shapes differ for ' + str(path))
            mapped = self.remember_mapping(key, array)
            # Keep small sources fast. Large sources remain file-backed without a
            # whole-volume float32 conversion or a second normalized-volume cache.
            if array.nbytes <= self.budget:
                while self.cache and self.bytes + array.nbytes > self.budget:
                    _, old = self.cache.popitem(last=False)
                    self.bytes -= old.nbytes
                array = np.array(array, copy=True)
                array.flags.writeable = False
                self.cache[key] = array
                self.bytes += array.nbytes
            if mapped:
                return array
            # Raster codecs need one full decode; spill once, never decode every crop.
            self.check_disk(array.nbytes)
            disk = np.lib.format.open_memmap(destination, mode='w+', dtype=array.dtype, shape=array.shape)
            for region in blocks(spatial):
                self.check()
                disk[region] = array[region]
            disk.flush()
            del disk
            if key in self.cache:
                return self.cache[key]
            return np.load(destination, mmap_mode='r')

    def pairs(self, pairs):
        sources = []
        for image, mask in pairs:
            self.check()
            key = (str(Path(image).resolve()), str(Path(mask).resolve()))
            if key not in self.sources:
                self.sources[key] = Source(self, Path(image), Path(mask))
            sources.append(self.sources[key])
        return [LazyArray(s, False) for s in sources], [LazyArray(s, True) for s in sources]

    def analysis(self, source):
        key = (str(source.mask), repr(getattr(source, 'bounds', None)))
        with self.lock:
            if key in self.analysis_cache:
                self.analysis_cache.move_to_end(key)
                return self.analysis_cache[key][0]
            path = self.directory / (hashlib.sha256(repr(key).encode()).hexdigest() + '.statistics.npz')
            if path.exists():
                with np.load(path) as saved:
                    value = dict(objects={int(i): (int(n), lo.copy(), hi.copy()) for i, n, lo, hi
                                 in zip(saved['ids'], saved['counts'], saved['low'], saved['high'])},
                                 foreground=saved['foreground'])
            else:
                value = mask_statistics(source.raw(True), self.check)
                objects = value['objects']
                np.savez(path, ids=list(objects), counts=[v[0] for v in objects.values()],
                         low=[v[1] for v in objects.values()], high=[v[2] for v in objects.values()],
                         foreground=value['foreground'])
            size = value['foreground'].nbytes + len(value['objects']) * 384
            if size <= self.analysis_budget:
                while self.analysis_cache and self.analysis_bytes + size > self.analysis_budget:
                    _, (_, old_size) = self.analysis_cache.popitem(last=False)
                    self.analysis_bytes -= old_size
                self.analysis_cache[key] = (value, size)
                self.analysis_bytes += size
            return value

    def close(self):
        with self.lock:
            self.cache.clear()
            self.analysis_cache.clear()
            self.sources.clear()
            self.mapped.clear()
            shutil.rmtree(self.directory, ignore_errors=True)

    def spatial_split(self, images, labels, patch, fraction):
        source = labels[0].source
        eligible = [i for i, (n, p) in enumerate(zip(source.shape, patch)) if n >= 2 * p]
        if not eligible:
            raise ValueError('The single source is too small for separate training and validation regions '
                             'at this patch size. Supply another image/volume or use a smaller preset.')
        axis = max(eligible, key=lambda i: source.shape[i] / patch[i])
        length, size = source.shape[axis], patch[axis]
        cut = length - max(size, min(length - size, int(round(length * fraction))))
        train_region = [slice(0, n) for n in source.shape]
        val_region = list(train_region)
        train_region[axis], val_region[axis] = slice(0, cut), slice(cut, length)
        train = Domain(source, tuple(train_region))
        val = Domain(source, tuple(val_region))
        val.normalization_source = train
        self.update(message='Single-source spatial holdout: training %s, validation %s. '
                    'Patches stay within their regions; normalization is fitted on the training region only.'
                    % (train.bounds, val.bounds), info=dict(type='dataset', training_region=train.bounds,
                                                         validation_region=val.bounds))
        return [LazyArray(train, False)], [LazyArray(train, True)], [LazyArray(val, False)], [LazyArray(val, True)]


class Source:
    def __init__(self, store, image, mask):
        self.store, self.image, self.mask = store, image, mask
        self.shape = None
        label = store.raw(self, True)
        self.shape = tuple(label.shape)
        self.statistics = None

    def raw(self, mask):
        return self.store.raw(self, mask)

    @property
    def analysis(self):
        return self.store.analysis(self)

    def fit_normalization(self):
        if self.statistics is not None:
            return self.statistics
        if hasattr(self, 'normalization_source'):
            return self.normalization_source.fit_normalization()
        image = self.raw(False)
        result = []
        for c in range(image.shape[-1]):
            channel = image[..., c]
            integer = channel.dtype.kind == 'u' and channel.dtype.itemsize <= 2
            histogram = np.zeros(np.iinfo(channel.dtype).max + 1, np.int64) if integer else None
            samples = []
            # Deterministic stratified float sampling; integer percentiles are exact.
            stride = max(1, int(math.ceil(channel.size / 262144)))
            offset = 0
            nonzero = False
            for region in blocks(self.shape):
                self.store.check()
                values = np.asarray(channel[region]).reshape(-1)
                if not np.all(np.isfinite(values)):
                    raise ValueError('Non-finite image pixels in ' + str(self.image))
                nonzero |= bool(np.any(values != 0))
                if integer:
                    histogram += np.bincount(values, minlength=len(histogram))
                else:
                    samples.append(values[(-offset) % stride::stride].astype(np.float32))
                    offset += len(values)
            if integer:
                cumulative = np.cumsum(histogram)
                ranks = (channel.size - 1) * np.array([0.01, 0.998])
                lo = np.searchsorted(cumulative, np.floor(ranks).astype(np.int64), side='right')
                hi = np.searchsorted(cumulative, np.ceil(ranks).astype(np.int64), side='right')
                low, high = lo + (hi - lo) * (ranks - np.floor(ranks))
            else:
                low, high = np.percentile(np.concatenate(samples), [1, 99.8])
            result.append((float(low), float(high - low) + 1e-20, nonzero))
        self.statistics = result
        self.store.update(message='Prepared cached normalization for ' + str(self.image),
                          info={'type': 'data_loading', 'path': str(self.image),
                                'normalization': 'exact_integer_or_bounded_float_percentiles'})
        return result

    def normalized(self, region):
        stats = self.fit_normalization()
        image = np.array(self.raw(False)[region], dtype=np.float32, copy=True)
        for c, (low, scale, nonzero) in enumerate(stats):
            if nonzero:
                image[..., c] -= low
                image[..., c] /= scale
            else:
                image[..., c] = 0
        channels = self.store.channels
        if image.shape[-1] == 1 and channels > 1:
            image = np.repeat(image, channels, axis=-1)
        elif image.shape[-1] < channels:
            image = np.concatenate((image, np.zeros(image.shape[:-1] + (channels - image.shape[-1],), np.float32)), axis=-1)
        return image[..., :channels]


class Domain(Source):
    def __init__(self, parent, region):
        self.parent, self.region = parent, region
        self.store, self.image, self.mask = parent.store, parent.image, parent.mask
        self.shape = tuple(s.stop - s.start for s in region)
        self.bounds = [[s.start, s.stop] for s in region]
        self.statistics = None

    def raw(self, mask):
        return self.parent.raw(mask)[self.region]


class LazyArray:
    def __init__(self, source, mask):
        self.source, self.mask = source, mask
        self.shape = source.shape + (() if mask else (source.store.channels,))
        self.ndim, self.size = len(self.shape), math.prod(self.shape)
        self.dtype = np.dtype(np.int32 if mask else np.float32)

    @property
    def analysis(self):
        return self.source.analysis

    def __getitem__(self, selection):
        self.source.store.check()
        if self.mask:
            return np.asarray(self.source.raw(True)[selection], dtype=np.int32)
        if not isinstance(selection, tuple):
            selection = (selection,)
        if any(s is Ellipsis for s in selection):
            index = next(i for i, s in enumerate(selection) if s is Ellipsis)
            selection = selection[:index] + (slice(None),) * (self.ndim - len(selection) + 1) + selection[index + 1:]
        selection += (slice(None),) * (self.ndim - len(selection))
        spatial = selection[:self.source.store.dimensions]
        result = self.source.normalized(spatial)
        if len(selection) > self.source.store.dimensions:
            result = result[(Ellipsis,) + selection[self.source.store.dimensions:]]
        return result

    def __array__(self, dtype=None, copy=None):
        # Used only by explicitly requested full-image operations, not patch training.
        estimate = self.size * self.dtype.itemsize
        available = available_memory()
        if available is not None and estimate > available * 0.4:
            raise MemoryError('Full-image operation exceeds the available memory budget.')
        result = self[(slice(None),) * self.ndim]
        return np.asarray(result, dtype=dtype)


def statistics(label):
    return label.analysis if isinstance(label, LazyArray) else mask_statistics(label)


def median_object_extents(labels, maximum=8192):
    """Estimate anisotropy without retaining every object's extent across the dataset."""
    from itertools import islice
    rng = np.random.RandomState(42)
    extents = np.empty((0, len(labels[0].shape)), np.int64)
    priorities = np.empty(0)
    for label in labels:
        objects = iter(statistics(label)['objects'].values())
        while True:
            if isinstance(label, LazyArray):
                label.source.store.check()
            chunk = list(islice(objects, 4096))
            if not chunk:
                break
            extents = np.concatenate((extents, [hi - lo for _, lo, hi in chunk]))
            priorities = np.concatenate((priorities, rng.random_sample(len(chunk))))
            if len(extents) > maximum:
                keep = np.argpartition(priorities, maximum - 1)[:maximum]
                extents, priorities = extents[keep], priorities[keep]
    return np.median(extents, axis=0) if len(extents) else None


def sample_start(label, patch, rng, foreground=False):
    if any(n < p for n, p in zip(label.shape, patch)):
        raise ValueError('StarDist source is smaller than train_patch_size: %s < %s' % (label.shape, patch))
    points = statistics(label)['foreground'] if foreground else ()
    if len(points):
        center = np.unravel_index(int(points[int(rng.randint(len(points)))]), label.shape)
        return tuple(int(rng.randint(max(0, c - p + 1), min(c, n - p) + 1))
                     for c, n, p in zip(center, label.shape, patch))
    return tuple(int(rng.randint(n - p + 1)) for n, p in zip(label.shape, patch))


def fingerprint(images, labels):
    result = []
    for image, label in zip(images, labels):
        item = {'shape': list(label.shape)}
        if isinstance(label, LazyArray):
            item['region'] = getattr(label.source, 'bounds', None)
            for key, path in [('image', label.source.image), ('mask', label.source.mask)]:
                info = path.stat()
                item[key] = [str(path.resolve()), info.st_size, info.st_mtime_ns]
        else:
            item['mask'] = hashlib.sha256(np.asarray(label).tobytes()).hexdigest()
        result.append(item)
    return result
