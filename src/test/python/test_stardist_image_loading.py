"""Exercise the exact generated StarDist loader without starting TensorFlow/training."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import tifffile


CODE = ast.parse(Path(sys.argv.pop(1)).read_text(encoding="utf-8"))
FUNCTIONS = {"_signature_format", "_read_format", "_read_array", "_physical_spacing", "_canonical_array"}
STATE = {"_reader_formats", "_reader_preferences", "_image_ome_metadata", "_reader_notices"}
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
