"""Dataset parity and numeric-image integration, without JDLL/Appose dependencies."""

import functools
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import random
import tempfile
import unittest

os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['MPLBACKEND'] = 'Agg'
sys.dont_write_bytecode = True

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset

MODULE = Path(__file__).resolve().parents[2] / 'main/resources/python/yolo_data.py'
spec = importlib.util.spec_from_file_location('jdll_yolo_data', MODULE)
data = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = data
spec.loader.exec_module(data)
NAMES = {0: 'cell', 1: 'nucleus'}


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def dataset(cls, root, augment=False, sources=None):
    kwargs = dict(img_path=str(root / 'images/train'), data={'names': NAMES}, task='detect', imgsz=128,
                  batch_size=2, cache=False, augment=augment, hyp=get_cfg(), rect=False, stride=32, pad=0)
    if sources is not None:
        kwargs['sources'] = sources
    return cls(**kwargs)


class YoloDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cv2.setNumThreads(1)

    def test_shuffled_epochs_match_native_including_fortran_and_duplicate_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for family in ('png', 'npy'):
                for kind in ('images', 'labels'):
                    (root / family / kind / 'train').mkdir(parents=True)
            for i in range(5):
                image = np.random.default_rng(i).integers(0, 256, (128, 160, 3), dtype=np.uint8)
                cv2.imwrite(str(root / 'png/images/train' / ('%d.png' % i)), image[..., ::-1])
                array = image.transpose(2, 0, 1)  # CYX, one of the agreed NumPy conventions.
                np.save(root / 'npy/images/train' / ('%d.npy' % i), np.asfortranarray(array) if i % 2 else array)
                label = '1 0.6 0.4 0.2 0.3\n0 0.3 0.7 0.1 0.2\n' if i < 4 else ''
                if i == 1:
                    label += '1 0.6 0.4 0.2 0.3\n'
                for family in ('png', 'npy'):
                    (root / family / 'labels/train' / ('%d.txt' % i)).write_text(label)
            fingerprints = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.npy')}
            seed(42)
            native = dataset(YOLODataset, root / 'png', True)
            seed(42)
            numpy = dataset(data.JdllYoloDataset, root / 'npy', True)
            for epoch in range(3):
                outputs = []
                for ds in (native, numpy):
                    seed(epoch + 7)
                    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0,
                                        generator=torch.Generator().manual_seed(epoch + 9), collate_fn=YOLODataset.collate_fn)
                    outputs.append(list(loader))
                self.assertEqual(len(outputs[0]), 3)
                for a, b in zip(*outputs):
                    self.assertEqual([Path(p).stem for p in a['im_file']], [Path(p).stem for p in b['im_file']])
                    for key in ('img', 'cls', 'bboxes', 'batch_idx'):
                        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
                self.assertEqual(outputs[0][-1]['img'].shape[0], 1)
            self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest() == sha for p, sha in fingerprints.items()))

    def test_float_training_statistics_virtual_crops_and_shared_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ('train', 'val'):
                for kind in ('images', 'labels'):
                    (root / kind / split).mkdir(parents=True)
            source = root / 'float.npy'
            image = np.linspace(10, 20, 128 * 160, dtype=np.float32).reshape(128, 160)
            np.save(source, image)
            validation = root / 'validation.tif'
            tifffile.imwrite(validation, np.linspace(-100, 1000, 128 * 128, dtype=np.float32).reshape(128, 128))
            train_id = root / 'images/train/sample.npy'
            val_id = root / 'images/val/sample.tif'
            entries = {str(train_id): {'source': str(source), 'crop': [16, 0, 128, 128], 'width': 128, 'height': 128},
                       str(val_id): {'source': str(validation), 'width': 128, 'height': 128}}
            (root / 'jdll_samples.json').write_text(json.dumps(entries))
            yaml = root / 'data.yaml'
            yaml.write_text(f'path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: cell\n  1: nucleus\n')
            for split in ('train', 'val'):
                (root / 'labels' / split / 'sample.txt').write_text('0 0.5 0.5 0.2 0.2\n')
            sources = data.Sources(yaml)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            normalization = sources.fit(str(root / 'images/train'), lambda **event: None)
            self.assertEqual(normalization['float_min'], float(image[:, 16:144].min()))
            self.assertEqual(normalization['float_max'], float(image[:, 16:144].max()))
            pixels = sources.pixels(train_id)
            self.assertEqual(pixels.dtype, np.uint8)
            self.assertEqual(pixels.shape, (128, 128, 3))
            self.assertEqual((int(pixels.min()), int(pixels.max())), (0, 255))
            self.assertEqual((int(sources.pixels(val_id).min()), int(sources.pixels(val_id).max())), (0, 255))
            for augment in (False, True):
                ds = dataset(data.JdllYoloDataset, root, augment, sources)
                if augment:
                    ds.close_mosaic(get_cfg())
                self.assertEqual(ds[0]['img'].dtype, torch.uint8)
            preview = root / 'preview'
            preview.mkdir()
            asset = data.preview_image(sources, train_id, preview)
            np.testing.assert_array_equal(cv2.imread(asset), pixels)
            self.assertFalse(train_id.exists(), 'Virtual samples must not create converted image files')
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            np.testing.assert_array_equal(data.prepare_pixels(np.array([0, 65535], np.uint16), normalization), [0, 255])
            np.testing.assert_array_equal(data.prepare_pixels(np.array([0, 65535], '>u2'), normalization), [0, 255])
            # Dispatch on content even when users supplied misleading extensions.
            disguised = root / 'float-as-png.png'
            disguised.write_bytes(validation.read_bytes())
            np.testing.assert_array_equal(sources.pixels(disguised), sources.pixels(val_id))
            raster = root / 'png-as-tiff.tif'
            raster.write_bytes(Path(asset).read_bytes())
            np.testing.assert_array_equal(sources.pixels(raster), pixels)
            np.testing.assert_array_equal(data.prepare_pixels(np.ones(2, np.float32), {'float_min': 1, 'float_max': 1}), [0, 0])
            with self.assertRaisesRegex(ValueError, 'NaN or Inf'):
                data.prepare_pixels(np.array([np.nan], np.float32), normalization)
            self.assertEqual(data.numpy_axes((128, 128, 512, 512)), 'ZCYX')
            self.assertEqual(data.numpy_axes((128, 160, 3)), 'XYC')
            with self.assertRaises(ValueError):
                data.numpy_axes((1, 1, 1, 1, 1, 1))
            # One real epoch exercises trainer setup, final validation and checkpoint
            # reload. Enabled explicitly; dataset parity tests need no training.
            if os.environ.get('JDLL_YOLO_SMOKE') == '1':
                from ultralytics import YOLO
                events = []
                trainer = data.trainer_class(yaml, None, lambda **event: events.append(event), lambda: False)
                model = YOLO('yolo26n.yaml')
                model.train(trainer=trainer, data=str(yaml), epochs=1, imgsz=128, batch=1, workers=0,
                            device='cpu', amp=False, plots=False, project=str(root / 'runs'), name='smoke',
                            cache=False, pretrained=False, verbose=False, close_mosaic=0)
                self.assertTrue((root / 'runs/smoke/weights/best.pt').is_file())
                self.assertTrue(any(e.get('info', {}).get('preprocessing') for e in events))
                self.assertEqual(model.trainer.jdll_sources.normalization, normalization)
                self.assertEqual(model.model.jdll_preprocessing, normalization)
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)


if __name__ == '__main__':
    unittest.main()
