"""Exercise the exact generated StarDist loader without starting TensorFlow/training."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import tifffile

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("numpy_images", Path(__file__).parents[2] / "main/resources/python/numpy_images.py")
numpy_images = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numpy_images)
data_spec = importlib.util.spec_from_file_location("stardist_data", Path(__file__).parents[2] / "main/resources/python/stardist_data.py")
stardist_data = importlib.util.module_from_spec(data_spec)
data_spec.loader.exec_module(stardist_data)

CODE = ast.parse(Path(sys.argv.pop(1)).read_text(encoding="utf-8"))
FUNCTIONS = {"_signature_format", "_read_format", "_read_array", "_physical_spacing", "_canonical_array",
             "_files", "_image_files", "_mask_key", "_pairs"}
STATE = {"_reader_formats", "_reader_preferences", "_image_ome_metadata", "_reader_notices", "IMAGE_EXTS"}
LOADER = ast.Module(body=[node for node in CODE.body if
    isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS or
    isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in STATE
                                        for target in node.targets for t in ast.walk(target))
], type_ignores=[])


class ImageLoadingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.messages = []
        self.ns = dict(np=np, Image=Image, TiffFile=tifffile.TiffFile, Path=Path,
                       TiffFileError=tifffile.TiffFileError, ET=ET, n_dim=3,
                       numpy_images=numpy_images, stardist_data=stardist_data,
                       _task_update=lambda **kw: self.messages.append(kw))
        exec(compile(LOADER, "<generated StarDist loader>", "exec"), self.ns)
        self.read = self.ns["_read_array"]
        self.labels = np.array([[0, 256, 65535], [1, 42, 1024]], dtype=np.uint16)

    def tiff(self, name, array=None, **kwargs):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, self.labels if array is None else array, **kwargs)
        return path

    def png(self, name, array=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.labels if array is None else array).save(path, format="PNG")
        return path

    def test_supported_formats_do_not_need_extra_signature_probes(self):
        for extension, fmt in [("png", "PNG"), ("bmp", "BMP"), ("jpg", "JPEG"), ("jpeg", "JPEG")]:
            for channels in [1, 3]:
                array = np.full((12, 13) if channels == 1 else (12, 13, 3), 72, np.uint8)
                path = self.root / f"image-{channels}.{extension}"
                Image.fromarray(array).save(path, format=fmt)
                with patch.dict(self.ns, _signature_format=lambda _: self.fail("Redundant header probe")):
                    result, axes = self.read(path)
                np.testing.assert_array_equal(result, array)
                self.assertEqual(axes, "YX" if channels == 1 else "YXC")
        for extension in ["tif", "tiff"]:
            path = self.tiff(f"labels.{extension}")
            with patch.dict(self.ns, _signature_format=lambda _: self.fail("Redundant header probe")):
                result, axes = self.read(path, is_mask=True)
            np.testing.assert_array_equal(result, self.labels)
            self.assertEqual(axes, "YX")

    def test_sixteen_bit_png_and_palette_mask_labels_are_preserved(self):
        result, _ = self.read(self.png("labels.png"), is_mask=True)
        np.testing.assert_array_equal(result, self.labels)
        indices = np.array([[0, 1, 2], [2, 1, 0]], np.uint8)
        palette = Image.fromarray(indices).convert("P")
        palette.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * 759)
        path = self.root / "palette.png"
        palette.save(path)
        result, axes = self.read(path, is_mask=True)
        np.testing.assert_array_equal(result, indices)
        self.assertEqual(axes, "YX")
        rgb, axes = self.read(path)
        np.testing.assert_array_equal(rgb, np.asarray(palette.convert("RGB")))
        self.assertEqual(axes, "YXC")

    def test_cache_is_scoped_and_mixed_contents_still_work(self):
        first = self.tiff("images/first.png")
        self.read(first)
        key = (str(first.parent), ".png", False)
        self.assertEqual(self.ns["_reader_preferences"][key], "TIFF")
        following = self.tiff("images/next.png")
        with patch.dict(self.ns, _signature_format=lambda _: self.fail("Cached reader was not used")):
            np.testing.assert_array_equal(self.read(following)[0], self.labels)
        for path, is_mask in [(self.png("masks/labels.png"), True), (self.png("another-dataset/input.png"), False),
                              (self.png("images/mask.png"), True)]:
            with patch.dict(self.ns, _signature_format=lambda _: self.fail("Reader preference leaked")):
                np.testing.assert_array_equal(self.read(path, is_mask=is_mask)[0], self.labels)
        actual_png = self.png("images/real.png")
        np.testing.assert_array_equal(self.read(actual_png)[0], self.labels)
        self.read(first)
        self.read(actual_png)
        self.assertEqual(len(self.messages), 2)  # One notice per discovered format, not per file.
        self.assertEqual(self.messages[0]["info"]["type"], "warning")

    def test_png_mislabeled_as_tiff_and_uppercase_extensions(self):
        first = self.png("images/first.TIFF")
        np.testing.assert_array_equal(self.read(first)[0], self.labels)
        following = self.png("images/next.tiff")
        with patch.dict(self.ns, _signature_format=lambda _: self.fail("Cached PNG reader was not used")):
            np.testing.assert_array_equal(self.read(following)[0], self.labels)
        np.testing.assert_array_equal(self.read(self.tiff("images/real.tiff"))[0], self.labels)

    def test_tiff_axes_precision_and_spacing_survive_wrong_extensions(self):
        array = np.arange(2 * 5 * 6, dtype=np.uint16).reshape(2, 5, 6)
        metadata = dict(axes="ZYX", PhysicalSizeZ=2.0, PhysicalSizeY=0.5, PhysicalSizeX=0.25)
        path = self.tiff("volume.png", array, ome=True, metadata=metadata, photometric="minisblack", compression="deflate")
        result, axes = self.read(path)
        np.testing.assert_array_equal(result, array)
        self.assertEqual(axes, "ZYX")
        with patch.dict(self.ns, TiffFile=lambda _: self.fail("Metadata should not reopen the TIFF")):
            self.assertEqual(self.ns["_physical_spacing"](path), (2.0, 0.5, 0.25))
        for bigtiff, byteorder in [(True, "<"), (True, ">"), (False, ">")]:
            path = self.tiff(f"volume-{bigtiff}-{byteorder == '>'}.png", array, bigtiff=bigtiff,
                             byteorder=byteorder, metadata={"axes": "ZYX"}, photometric="minisblack")
            np.testing.assert_array_equal(self.read(path)[0], array)
        floating = np.array([[0.01, 1.25], [-10.5, 999.0]], np.float32)
        result, _ = self.read(self.tiff("float.tif", floating))
        np.testing.assert_array_equal(result, floating)

    def test_corruption_or_decoder_failure_does_not_switch_readers(self):
        path = self.tiff("valid.tif")
        with patch.dict(self.ns, _read_format=lambda *args: (_ for _ in ()).throw(ValueError("missing codec"))):
            with self.assertRaisesRegex(OSError, "valid.tif.*missing codec"):
                self.read(path)
        self.assertFalse(self.ns["_reader_preferences"])
        path = self.root / "broken.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\ntruncated")
        with self.assertRaisesRegex(OSError, "broken.png"):
            self.read(path)
        path = self.root / "garbage.tif"
        path.write_bytes(b"not an image")
        with self.assertRaisesRegex(OSError, "garbage.tif"):
            self.read(path)
        self.assertFalse(self.ns["_reader_preferences"])

    def test_failed_fallback_does_not_poison_cache(self):
        path = self.root / "broken.png"
        path.write_bytes(b"II\x2a\x00\x08\x00\x00\x00truncated")
        with self.assertRaisesRegex(OSError, "broken.png"):
            self.read(path)
        self.assertFalse(self.ns["_reader_preferences"])
        self.assertFalse(self.messages)

    def test_jpeg_masks_rejected_even_with_png_extension(self):
        for extension in ["jpg", "png"]:
            path = self.root / f"mask.{extension}"
            Image.fromarray(np.zeros((12, 13), np.uint8)).save(path, format="JPEG")
            with self.assertRaisesRegex(OSError, "JPEG cannot be used"):
                self.read(path, is_mask=True)
        self.assertFalse(self.ns["_reader_preferences"])

    def test_multiple_series_and_animated_png_are_not_silently_truncated(self):
        path = self.root / "multiple.tif"
        with tifffile.TiffWriter(path) as writer:
            writer.write(self.labels)
            writer.write(np.zeros((10, 12), np.uint16))
        with self.assertRaisesRegex(OSError, "Multiple TIFF series"):
            self.read(path)
        path = self.root / "animated.png"
        Image.new("RGB", (12, 13), "red").save(path, save_all=True,
            append_images=[Image.new("RGB", (12, 13), "blue")], duration=100, loop=0)
        with self.assertRaisesRegex(OSError, "Multiple raster frames"):
            self.read(path)

    def test_numpy_images_and_masks_stay_mapped_and_reuse_patch_normalization(self):
        for dimensions, axes in ((2, "CYX"), (2, "XYC"), (3, "ZCYX"), (3, "XYCZ")):
            self.ns["n_dim"] = dimensions
            canonical_axes = "YXC" if dimensions == 2 else "ZYXC"
            shape = (128, 160, 2) if dimensions == 2 else (4, 128, 160, 2)
            image = np.random.default_rng(2).normal(2000, 100, shape).astype(np.float32)
            labels = np.zeros(shape[:-1], np.uint32)
            labels[..., 30:50, 50:70] = 70000
            image_path = self.root / f"input-{dimensions}-{axes}.npy"
            mask_path = self.root / f"mask-{dimensions}-{axes}.npy"
            np.save(image_path, np.asfortranarray(image.transpose(tuple(canonical_axes.index(axis) for axis in axes))))
            np.save(mask_path, labels.astype(">u4"))
            raw, stored_axes = self.read(image_path)
            np.testing.assert_array_equal(self.ns["_canonical_array"](raw, stored_axes, False), image)
            store = stardist_data.DatasetStore(self.read, self.ns["_canonical_array"], dimensions, 2,
                    {"cache_mb": 0}, lambda **event: None, lambda: False)
            try:
                inputs, targets = store.pairs([(image_path, mask_path)])
                selection = (slice(1, 3), slice(12, 42), slice(10, 45)) if dimensions == 3 else (slice(12, 42), slice(10, 45))
                np.testing.assert_array_equal(targets[0][selection], labels[selection])
                self.assertEqual(inputs[0][(*selection, slice(None))].shape, tuple(s.stop - s.start for s in selection) + (2,))
                self.assertEqual({Path(value[0]) for value in store.mapped.values()}, {image_path, mask_path})
                self.assertFalse(list(store.directory.glob("*.npy")), "NumPy sources must not be copied into the decoded cache")
                self.assertFalse(np.load(image_path, mmap_mode="r").flags.writeable)
            finally:
                store.close()

    def test_invalid_numpy_rank_objects_and_time_axes_name_the_source(self):
        path = self.root / "unsupported.npy"
        for array in (np.empty((1,) * 6), np.array([[object()]], object), np.zeros((3, 4, 1, 128, 160), np.uint8)):
            np.save(path, array)
            with self.assertRaisesRegex(OSError, "unsupported.npy"):
                self.read(path)

    def test_auto_cache_limit_and_explicit_overrides(self):
        for available, expected in ((None, 64 * 1024**2), (0, 0), (10 * 1024**3, 1024**3),
                                    (20 * 1024**3, 2 * 1024**3), (64 * 1024**3, 2 * 1024**3)):
            with patch.object(stardist_data, 'available_memory', return_value=available):
                self.assertEqual(expected, stardist_data.cache_bytes('auto'))
        self.assertEqual(0, stardist_data.cache_bytes(0))
        self.assertEqual(32 * 1024**2, stardist_data.cache_bytes(32))
        for invalid in (-1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'cache_mb'):
                stardist_data.cache_bytes(invalid)

    def store(self, dimensions, channels=1, budget=0):
        self.ns['n_dim'] = dimensions
        store = stardist_data.DatasetStore(self.read, self.ns['_canonical_array'], dimensions, channels,
                {'cache_mb': budget, 'cache_dir': str(self.root)}, lambda **event: self.messages.append(event), lambda: False)
        self.addCleanup(store.close)
        self.ns['storage'] = store
        return store

    def test_compressed_tiff_crops_preserve_axes_values_and_normalization(self):
        for dimensions, axes in ((2, 'YXC'), (2, 'CYX'), (2, 'TCYX'), (2, 'XYC'),
                                  (3, 'ZYXC'), (3, 'CZYX'), (3, 'TZCYX'), (3, 'XYZC')):
            with self.subTest(axes=axes):
                canonical_axes = 'YXC' if dimensions == 2 else 'ZYXC'
                shape = (32, 48, 2) if dimensions == 2 else (4, 32, 48, 2)
                image = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
                image[..., 1] = 0  # Zero-channel normalization must stay unchanged.
                expanded = image[None] if 'T' in axes else image
                native_axes = ('T' if 'T' in axes else '') + canonical_axes
                native = expanded.transpose(tuple(native_axes.index(a) for a in axes))
                labels = np.zeros(shape[:-1], np.uint32)
                labels[..., 8:20, 12:30] = 70000
                # First mask channel is used, irrespective of image channel layout.
                mask = np.stack((labels, np.ones_like(labels)), axis=-1)
                path = self.tiff('image-%s.png' % axes, native, compression='deflate',
                                 photometric='minisblack', metadata={'axes': axes})
                mask_path = self.tiff('mask-%s.tif' % axes, mask, compression='deflate',
                                      photometric='minisblack', metadata={'axes': canonical_axes})
                store = self.store(dimensions, channels=3)
                with patch.object(tifffile.TiffPageSeries, 'asarray', side_effect=AssertionError('Whole TIFF decode')):
                    x, y = store.pairs([(path, mask_path)])
                    region = (slice(1, 3), slice(5, 21), slice(7, 28)) if dimensions == 3 else (slice(5, 21), slice(7, 28))
                    np.testing.assert_array_equal(y[0][region], labels[region])
                    normalized = x[0][region]
                low, high = np.percentile(image[..., 0], [1, 99.8])
                expected = image[region].astype(np.float32)
                expected[..., 0] = (expected[..., 0] - low) / (high - low + 1e-20)
                expected = np.concatenate((expected, np.zeros(expected.shape[:-1] + (1,), np.float32)), axis=-1)
                np.testing.assert_allclose(normalized, expected, rtol=1e-6, atol=1e-6)
                self.assertEqual(0, store.bytes)
                self.assertFalse(list(store.directory.glob('*.npy')))
                self.assertFalse(list(store.directory.glob('*.decoded')))
                store.close()

    def test_tiled_and_stripped_tiffs_decode_only_selected_regions(self):
        image = np.arange(128 * 160, dtype=np.uint16).reshape(128, 160)
        for options in ({'tile': (32, 32)}, {'rowsperstrip': 8}):
            path = self.tiff('regions-%s.tif' % len(options.get('tile', ())), image,
                             compression='deflate', photometric='minisblack', **options)
            store = self.store(2)
            with patch.object(tifffile.TiffPageSeries, 'asarray', side_effect=AssertionError('Whole TIFF decode')):
                with tifffile.TiffFile(path) as tif:
                    raw = store.tiff_array(path, tif.series[0])
                region = (slice(37, 54), slice(41, 62))
                with patch.object(store, 'read_tiff_region', wraps=store.read_tiff_region) as read:
                    np.testing.assert_array_equal(raw[region], image[region])
                self.assertEqual(tuple(slice(s.start, s.stop, 1) for s in region), read.call_args.args[1])
            self.assertFalse(list(store.directory.iterdir()))

    def test_float_normalization_and_ignored_masks_match_cached_full_image_loading(self):
        rng = np.random.RandomState(7)
        image = rng.normal(500, 100, (4, 256, 384, 2)).astype(np.float32)
        labels = np.zeros(image.shape[:-1], np.int32)
        labels[:, 8:20, 12:30] = 70000
        labels[:, :2] = -1
        image_path = self.tiff('float-planar.tif', image.transpose(3, 0, 1, 2), compression='deflate',
                               photometric='minisblack', metadata={'axes': 'CZYX'})
        mask_path = self.tiff('ignored.tif', labels, compression='deflate',
                              photometric='minisblack', metadata={'axes': 'ZYX'})
        region = (slice(1, 3), slice(0, 32), slice(7, 28))
        cached = self.store(3, channels=2, budget=16)
        x, y = cached.pairs([(image_path, mask_path)])
        expected_image, expected_labels = x[0][region], y[0][region]
        expected_statistics = x[0].source.fit_normalization()
        lazy = self.store(3, channels=2)
        x, y = lazy.pairs([(image_path, mask_path)])
        np.testing.assert_array_equal(x[0][region], expected_image)
        np.testing.assert_array_equal(y[0][region], expected_labels)
        self.assertEqual(expected_statistics, x[0].source.fit_normalization())
        self.assertEqual(0, lazy.bytes)

    def test_pyramidal_tiff_reads_the_same_full_resolution_level_as_before(self):
        image = np.arange(128 * 160, dtype=np.uint16).reshape(128, 160)
        path = self.root / 'pyramid.tif'
        with tifffile.TiffWriter(path) as writer:
            writer.write(image, subifds=1, tile=(32, 32), compression='deflate', photometric='minisblack')
            writer.write(image[::2, ::2], subfiletype=1, tile=(32, 32), compression='deflate', photometric='minisblack')
        store = self.store(2)
        with patch.object(tifffile.TiffPageSeries, 'asarray', side_effect=AssertionError('Whole pyramid decode')):
            x, y = store.pairs([(path, path)])
            self.assertEqual(image.shape, y[0].shape)
            np.testing.assert_array_equal(y[0][37:54, 41:62], image[37:54, 41:62])
        self.assertEqual(0, store.bytes)

    def test_region_array_composed_views_match_numpy_without_decoding_unused_pixels(self):
        native = np.arange(2 * 3 * 8 * 9, dtype=np.uint16).reshape(2, 3, 8, 9)
        selections = []
        def read(selection):
            selections.append(selection)
            return native[selection]
        lazy = stardist_data.RegionArray(native.shape, native.dtype, read)
        actual = lazy[1, :, 2:8, 1:9:2].transpose((1, 2, 0))[1:4, :2, :2].expand_dims()
        expected = native[1, :, 2:8, 1:9:2].transpose(1, 2, 0)[1:4, :2, :2, None]
        self.assertFalse(selections)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual((1, slice(0, 2, 1), slice(3, 6, 1), slice(1, 5, 2)), selections[0])
        np.testing.assert_array_equal(lazy[..., None], native[..., None])
        np.testing.assert_array_equal(lazy[:, :, ::-1], native[:, :, ::-1])

    def test_png_ram_cache_is_bounded_reused_and_never_spilled_to_disk(self):
        image = np.full((12, 13), 72, np.uint8)
        mask = np.ones((12, 13), np.uint16)
        image_path, mask_path = self.png('input.png', image), self.png('target.png', mask)
        store = self.store(2, budget=1)
        with patch.dict(self.ns, _read_format=unittest.mock.Mock(wraps=self.ns['_read_format'])):
            x, y = store.pairs([(image_path, mask_path)])
            _ = x[0][2:5, 3:7]
            calls = self.ns['_read_format'].call_count
            for _ in range(3):
                _ = x[0][2:5, 3:7], y[0][2:5, 3:7]
            self.assertEqual(calls, self.ns['_read_format'].call_count)
        self.assertLessEqual(store.bytes, store.budget)
        # Eviction never writes an image cache, and a later full decode can be cached again.
        for i in range(3):
            source = type('Source', (), {'image': self.png('extra%d.png' % i, np.zeros((800, 800), np.uint8)),
                                         'mask': mask_path, 'shape': (800, 800)})()
            store.raw(source, False)
            self.assertLessEqual(store.bytes, store.budget)
        np.testing.assert_array_equal(y[0][2:5, 3:7], mask[2:5, 3:7])
        self.assertTrue(all(not array.flags.writeable for array in store.cache.values()))
        self.assertFalse(list(store.directory.iterdir()))

    def test_tiff_region_backend_fallback_is_explicit_and_memory_guarded(self):
        path = self.tiff('compressed.tif', np.ones((32, 48), np.uint16), compression='deflate')
        store = self.store(2)
        with tifffile.TiffFile(path) as tif:
            raw = store.tiff_array(path, tif.series[0])
        with patch.dict(sys.modules, zarr=None):
            np.testing.assert_array_equal(raw[2:5, 3:7], np.ones((3, 4), np.uint16))
            np.testing.assert_array_equal(raw[2:5, 3:7], np.ones((3, 4), np.uint16))
            with patch.object(stardist_data, 'available_memory', return_value=1):
                with self.assertRaisesRegex(MemoryError, 'compressed.tif'):
                    np.asarray(raw[2:5, 3:7])
        self.assertEqual(1, len([e for e in self.messages if e.get('info', {}).get('type') == 'warning']))
        self.assertFalse(list(store.directory.iterdir()))

    def test_uncompressed_tiff_maps_original_pixels_and_remains_read_only(self):
        image = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
        path = self.tiff('uncompressed.tif', image, photometric='minisblack')
        store = self.store(2)
        x, y = store.pairs([(path, path)])
        np.testing.assert_array_equal(y[0][2:5, 3:7], image[2:5, 3:7])
        _ = x[0][2:5, 3:7]
        self.assertEqual({path}, {Path(value[0]) for value in store.mapped.values()})
        self.assertFalse(list(store.directory.iterdir()))
        np.testing.assert_array_equal(tifffile.imread(path), image)

    def test_raster_images_take_precedence_over_numpy_caches_without_filtering_masks(self):
        image = self.tiff("images/case.tif")
        (image.parent / "case.npy").write_bytes(b"stale cache")
        np.save(image.parent / "other.npy", np.ones((128, 160), np.uint8))
        self.tiff("masks/case_mask.tif")
        self.tiff("masks/other_mask.tif")
        pairs = self.ns["_pairs"](image.parent, self.root / "masks")
        self.assertEqual([pair[0].name for pair in pairs], ["case.tif", "other.npy"])
        np.save(self.root / "masks/case_mask.npy", np.ones((128, 160), np.uint16))
        self.assertEqual(len(self.ns["_files"](self.root / "masks")), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
