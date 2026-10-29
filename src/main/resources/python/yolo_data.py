"""Read-only numeric images for YOLO; normalization is applied to RAM samples only."""

import hashlib
import json
import math
from contextlib import contextmanager
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils.ops import segments2boxes

DOCUMENTATION = 'https://github.com/bioimage-io/JDLL/blob/main/docs/image-formats.md'


def invalid(path, reason):
    return ValueError('%s: %s. See %s' % (reason, path, DOCUMENTATION))


def numpy_axes(shape):
    if not 2 <= len(shape) <= 5 or not all(shape):
        raise ValueError('Unsupported image type: expected 2 to 5 nonempty dimensions')
    first = ('XY', 'XYC', 'XYCZ', 'XYCZT')[len(shape) - 2]
    last = ('YX', 'CYX', 'ZCYX', 'TZCYX')[len(shape) - 2]
    candidates = []
    for axes in (first, last):
        x, y = shape[axes.index('X')], shape[axes.index('Y')]
        if min(x, y) > 100 and max(x, y) / min(x, y) < 3:
            candidates.append((x * y, axes == last, axes))
    if not candidates:
        raise ValueError('Cannot infer XY/C/Z/T axes from %s' % (shape,))
    return max(candidates)[2]


def geometry(shape, axes):
    axes = axes.upper().replace('S', 'C')
    if len(axes) != len(shape) or len(set(axes)) != len(axes):
        raise ValueError('Ambiguous image axes %s for %s' % (axes, shape))
    for axis, size in zip(axes, shape):
        if axis not in 'XYC' and size != 1:
            raise ValueError('YOLO requires 2D samples, not non-singleton %s axes' % axis)
    channels = shape[axes.index('C')] if 'C' in axes else 1
    if channels not in (1, 3, 4):
        raise ValueError('YOLO images require one or three channels (RGBA alpha is discarded)')
    return (shape[axes.index('Y')], shape[axes.index('X')], channels)


def canonical(array, axes):
    axes = axes.upper().replace('S', 'C')
    geometry(array.shape, axes)
    for i in range(len(axes) - 1, -1, -1):
        if axes[i] not in 'XYC':
            selection = [slice(None)] * array.ndim
            selection[i] = 0
            array = array[tuple(selection)]
            axes = axes[:i] + axes[i + 1:]
    target = 'YXC' if 'C' in axes else 'YX'
    array = array.transpose(tuple(axes.index(axis) for axis in target))
    return array[..., None] if array.ndim == 2 else array[..., :3]


def load_normalization(weights):
    for directory in (Path(weights).resolve().parent, *Path(weights).resolve().parent.parents):
        config = directory / 'config.json'
        if config.is_file():
            with config.open() as stream:
                content = json.load(stream)
            return content.get('preprocessing') if content.get('framework') == 'yolo' else None
    return None


def prepare_pixels(array, normalization):
    """Return byte-scale pixels without ever modifying the supplied array."""
    if array.dtype == np.uint8:
        return array
    if array.dtype.kind == 'u' and array.dtype.itemsize == 2:
        return np.right_shift(array, 8).astype(np.uint8)
    if array.dtype.kind == 'b':
        return array.astype(np.uint8) * 255
    if array.dtype.kind != 'f':
        raise ValueError('Unsupported image dtype %s; use uint8, uint16, or floating-point images' % array.dtype)
    if not np.isfinite(array).all():
        raise ValueError('Floating-point images must not contain NaN or Inf')
    low = None if normalization is None else normalization.get('float_min')
    high = None if normalization is None else normalization.get('float_max')
    if low is None or high is None:
        # Legacy/pretrained models have no training statistics. Their current
        # image is the only available reference; trained JDLL models never refit.
        low, high = float(array.min()), float(array.max())
    if high == low:
        return np.zeros(array.shape, dtype=np.uint8)
    if not math.isfinite(high - low):
        raise ValueError('Floating-point intensity range is too large; rescale the source values')
    working = array.astype(np.float64, copy=True)
    working -= low
    working /= high - low
    np.clip(working, 0, 1, out=working)
    working *= 255
    np.rint(working, out=working)
    return working.astype(np.uint8)


class Sources:
    def __init__(self, dataset_yaml=None, normalization=None, check=lambda: None):
        self.normalization, self.check = normalization, check
        self.entries, self.metadata = {}, {}
        if dataset_yaml:
            manifest = Path(dataset_yaml).resolve().parent / 'jdll_samples.json'
            if manifest.is_file():
                self.entries = json.loads(manifest.read_text())

    def entry(self, path):
        return self.entries.get(str(Path(path).absolute()), {'source': str(path)})

    def info(self, path):
        key = str(path)
        if key not in self.metadata:
            self.check()
            entry = self.entry(path)
            source = Path(entry['source'])
            try:
                with source.open('rb') as stream:
                    magic = stream.read(8)
                kind = 'npy' if magic.startswith(b'\x93NUMPY') else 'tiff' if magic[:4] in (
                    b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+') else 'raster'
                if kind == 'npy':
                    array = np.load(source, mmap_mode='r', allow_pickle=False)
                    try:
                        axes, dtype = numpy_axes(array.shape), array.dtype
                        height, width, channels = geometry(array.shape, axes)
                    finally:
                        array._mmap.close()
                elif kind == 'tiff':
                    with tifffile.TiffFile(source) as tif:
                        if len(tif.series) != 1:
                            raise ValueError('Multiple TIFF series; export one series')
                        series = tif.series[0]
                        axes, dtype = series.axes, series.dtype
                        height, width, channels = geometry(series.shape, axes)
                else:
                    with Image.open(source) as image:
                        width, height = image.size
                        if image.getexif().get(274) in (5, 6, 7, 8):
                            width, height = height, width
                        axes, dtype, channels = 'YXC', np.dtype('uint8'), 3
                if entry.get('crop'):
                    x, y, w, h = entry['crop']
                    if min(x, y) < 0 or min(w, h) < 1 or x + w > width or y + h > height:
                        raise ValueError('Crop is outside the source image')
                    width, height = w, h
                self.metadata[key] = (height, width, channels, np.dtype(dtype), axes, kind)
            except Exception as exc:
                if isinstance(exc, InterruptedError):
                    raise
                raise invalid(source, str(exc)) from exc
        return self.metadata[key]

    @contextmanager
    def image(self, path):
        self.check()
        source = Path(self.entry(path)['source'])
        mapping = None
        try:
            axes, kind = self.info(path)[4:]
            rgb = kind in ('npy', 'tiff')
            if kind == 'npy':
                mapping = np.load(source, mmap_mode='r', allow_pickle=False)
                array = canonical(mapping, axes)
            elif kind == 'tiff':
                try:
                    mapping = tifffile.memmap(source, mode='r')
                    array = canonical(mapping, axes)
                except ValueError:
                    array = canonical(tifffile.imread(source), axes)
            else:
                array = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if array is None:
                    raise ValueError('Cannot decode image')
                array = array[..., None] if array.ndim == 2 else array[..., :3]
            crop = self.entry(path).get('crop')
            if crop:
                x, y, width, height = crop
                array = array[y:y + height, x:x + width]
            yield array, rgb
        except Exception as exc:
            if isinstance(exc, InterruptedError):
                raise
            raise invalid(source, str(exc)) from exc
        finally:
            if mapping is not None:
                mapping._mmap.close()

    def pixels(self, path, channels=3):
        with self.image(path) as (array, rgb):
            array = prepare_pixels(array, self.normalization)
            if channels == 1 and array.shape[-1] == 3:
                array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY)[..., None]
            elif channels == 3 and array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            elif channels != array.shape[-1]:
                raise invalid(path, 'Image/model channel counts differ')
            if rgb and channels == 3:
                array = array[..., ::-1]
            return np.array(array, order='C', copy=True)

    def files(self, paths):
        roots = paths if isinstance(paths, (list, tuple)) else [paths]
        result = set()
        for root in roots:
            path = Path(root).absolute()
            if path.is_dir():
                files = (p for p in path.rglob('*') if p.is_file())
                result.update(str(p) for p in files if p.suffix.lower().lstrip('.') in IMG_FORMATS | {'npy'})
                result.update(p for p in self.entries if Path(p).is_relative_to(path))
            elif path.suffix.lower() == '.txt':
                for line in path.read_text(encoding='utf-8-sig').splitlines():
                    if line.strip() and not line.lstrip().startswith('#'):
                        p = Path(line.strip())
                        result.add(str(p if p.is_absolute() else path.parent / p))
            else:
                raise invalid(path, 'Expected an image directory or image list')
        ordinary = {str(Path(p).with_suffix('')) for p in result if not p.lower().endswith('.npy')}
        return sorted(p for p in result if not (p.lower().endswith('.npy') and str(Path(p).with_suffix('')) in ordinary))

    def fit(self, paths, notify, fraction=1.0):
        files = self.files(paths)
        files = files[:max(1, round(len(files) * fraction))]
        signature = hashlib.sha256()
        for path in files:
            self.check()
            source = Path(self.entry(path)['source'])
            stat = source.stat()
            signature.update(repr((str(source.resolve()), stat.st_size, stat.st_mtime_ns, self.entry(path).get('crop'))).encode())
        signature = signature.hexdigest()
        if self.normalization and self.normalization.get('training_signature') == signature:
            return self.normalization
        low, high, seen = math.inf, -math.inf, set()
        for i, path in enumerate(files):
            if self.info(path)[3].kind != 'f':
                continue
            entry = self.entry(path)
            identity = (entry['source'], tuple(entry.get('crop', ())))
            if identity in seen:
                continue
            seen.add(identity)
            with self.image(path) as (array, _):
                iterator = np.nditer(array, flags=['external_loop', 'buffered'], order='K', buffersize=262144)
                for block in iterator:
                    self.check()
                    if not np.isfinite(block).all():
                        raise invalid(path, 'Floating-point training images contain NaN or Inf')
                    low, high = min(low, float(block.min())), max(high, float(block.max()))
            if i % max(1, len(files) // 20) == 0 or i + 1 == len(files):
                notify(message='Analyzing floating-point training intensities: %d/%d images.' % (i + 1, len(files)),
                       info={'type': 'configuration', 'phase': 'normalization', 'current': i + 1, 'maximum': len(files)})
        self.normalization = {'kind': 'dataset_min_max', 'float_min': low if seen else None,
                              'float_max': high if seen else None, 'uint16': 'high_byte',
                              'training_signature': signature}
        return self.normalization


class JdllYoloDataset(YOLODataset):
    def __init__(self, *args, sources=None, **kwargs):
        self.sources = sources or Sources()
        if kwargs.get('cache'):
            raise ValueError('JDLL numeric datasets use read-only sources; use cache=False')
        super().__init__(*args, **kwargs)

    def get_img_files(self, img_path):
        files = self.sources.files(img_path)
        if not files:
            raise invalid(img_path, 'No training images found')
        return files[:max(1, round(len(files) * self.fraction))]

    def get_labels(self):
        self.label_files = img2label_paths(self.im_files)
        labels = []
        for image, label in zip(self.im_files, self.label_files):
            self.sources.check()
            height, width = self.sources.info(image)[:2]
            try:
                rows = [line.split() for line in Path(label).read_text().splitlines() if line.strip()] if Path(label).exists() else []
                segments = []
                if any(len(row) > 6 for row in rows):
                    segments = [np.asarray(row[1:], dtype=np.float32).reshape(-1, 2) for row in rows]
                    values = np.concatenate((np.asarray([row[0] for row in rows], dtype=np.float32)[:, None], segments2boxes(segments)), axis=1)
                else:
                    if any(len(row) != 5 for row in rows):
                        raise ValueError('Expected class x_center y_center width height')
                    values = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
                if len(values):
                    if not np.isfinite(values).all() or values[:, 1:].min() < -0.01 or values[:, 1:].max() > 1.01:
                        raise ValueError('Invalid normalized box coordinates')
                    cls = values[:, 0]
                    if (cls < 0).any() or (cls >= len(self.data['names'])).any() or (cls != np.floor(cls)).any():
                        raise ValueError('Invalid class IDs')
                    _, indices = np.unique(values, axis=0, return_index=True)
                    if len(indices) < len(values):
                        values = values[indices]
                        if segments:
                            segments = [segments[i] for i in indices]
                labels.append(dict(im_file=image, shape=(height, width), cls=values[:, :1], bboxes=values[:, 1:],
                                   segments=segments, keypoints=None, normalized=True, bbox_format='xywh'))
            except Exception as exc:
                raise invalid(label, str(exc)) from exc
        return labels

    def load_image(self, i, rect_mode=True):
        if self.ims[i] is not None:
            return self.ims[i], self.im_hw0[i], self.im_hw[i]
        image = self.sources.pixels(self.im_files[i], self.channels)
        h, w = image.shape[:2]
        if rect_mode:
            ratio = self.imgsz / max(h, w)
            if ratio != 1:
                image = cv2.resize(image, (min(math.ceil(w * ratio), self.imgsz), min(math.ceil(h * ratio), self.imgsz)), interpolation=cv2.INTER_LINEAR)
        elif (h, w) != (self.imgsz, self.imgsz):
            image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        if image.ndim == 2:
            image = image[..., None]
        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = image, (h, w), image.shape[:2]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                old = self.buffer.pop(0)
                self.ims[old], self.im_hw0[old], self.im_hw[old] = None, None, None
        return image, (h, w), image.shape[:2]

    def cache_images_to_disk(self, i):
        raise ValueError('NumPy files are source data, never writable Ultralytics caches')


def trainer_class(dataset_yaml, cached_normalization, notify, cancelled):
    def check():
        if cancelled():
            raise InterruptedError('YOLO dataset analysis cancelled')

    sources = Sources(dataset_yaml, cached_normalization, check)

    def dataset(args, path, batch, data, mode, stride):
        return JdllYoloDataset(img_path=path, imgsz=args.imgsz, batch_size=batch, augment=mode == 'train',
                               hyp=copy(args), rect=args.rect or mode == 'val', cache=False,
                               single_cls=args.single_cls, stride=stride, pad=0 if mode == 'train' else 0.5,
                               task=args.task, classes=args.classes, data=data,
                               fraction=args.fraction if mode == 'train' else 1.0, sources=sources)

    class Validator(DetectionValidator):
        def build_dataset(self, img_path, mode='val', batch=None):
            return dataset(self.args, img_path, batch, self.data, mode, self.stride)

    class Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            model.jdll_preprocessing = sources.normalization
            return model

        def get_dataset(self):
            if self.args.cache:
                raise ValueError('JDLL read-only datasets require cache=False; source NumPy files are not image caches')
            data = super().get_dataset()
            self.jdll_sources = sources
            normalization = sources.fit(data['train'], notify, self.args.fraction)
            data['jdll_preprocessing'] = normalization
            notify(message='YOLO input preprocessing: ' + json.dumps(normalization),
                   info={'type': 'runtime', 'preprocessing': normalization})
            return data

        def build_dataset(self, img_path, mode='train', batch=None):
            stride = max(int(self.model.stride.max()), 32)
            return dataset(self.args, img_path, batch, self.data, mode, stride)

        def get_validator(self):
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
            return Validator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    return Trainer


def preview_image(sources, path, directory):
    """Only small display assets are written; training source pixels stay untouched."""
    destination = Path(directory) / (hashlib.sha256(str(path).encode()).hexdigest() + '.png')
    if not destination.exists():
        image = sources.pixels(path)
        ratio = min(1.0, 1024 / max(image.shape[:2]))
        if ratio < 1:
            image = cv2.resize(image, (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))))
        if not cv2.imwrite(str(destination), image):
            raise OSError('Cannot save validation preview ' + str(destination))
    return str(destination)
