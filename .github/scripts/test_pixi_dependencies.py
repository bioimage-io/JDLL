"""Distinguish one conda-forge OpenCV provider from conflicting OpenCV wheels."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from smoke_jdll_models import shared_conda_opencv_metadata


class OpenCVProviderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.prefix = Path(temporary.name).resolve()
        self.site = self.prefix / "lib/python3.11/site-packages"
        self.installers = {"opencv-python": "conda", "opencv-python-headless": "conda"}
        self.record = {
            "name": "py-opencv", "version": "4.13.0", "build": "headless_test_0",
            "files": ["lib/python3.11/site-packages/cv2/__init__.py",
                      "lib/python3.11/site-packages/opencv_python-4.13.0.dist-info/METADATA",
                      "lib/python3.11/site-packages/opencv_python_headless-4.13.0.dist-info/METADATA"],
        }
        self.libraries = {"name": "libopencv", "version": "4.13.0", "build": "headless_test_0", "files": []}
        (self.prefix / "conda-meta").mkdir()

    def check(self):
        (self.prefix / "conda-meta/py-opencv-test.json").write_text(json.dumps(self.record))
        (self.prefix / "conda-meta/libopencv-test.json").write_text(json.dumps(self.libraries))

        def distribution(name):
            return SimpleNamespace(version="4.13.0", read_text=lambda key: self.installers[name],
                                   locate_file=lambda path: self.site / path)

        with patch("smoke_jdll_models.sys.prefix", str(self.prefix)), \
                patch("smoke_jdll_models.metadata.distribution", side_effect=distribution), \
                patch("smoke_jdll_models.importlib.util.find_spec",
                      return_value=SimpleNamespace(origin=str(self.site / "cv2/__init__.py"))):
            return shared_conda_opencv_metadata()

    def test_single_headless_conda_provider_with_both_metadata_names_is_valid(self):
        self.assertTrue(self.check())

    def test_split_conda_packages_can_own_the_metadata_and_bindings_separately(self):
        self.libraries["files"] = self.record["files"][1:]
        self.record["files"] = self.record["files"][:1]
        self.assertTrue(self.check())

    def test_wheel_overwriting_a_conda_install_is_rejected(self):
        self.installers["opencv-python"] = "uv"
        self.assertFalse(self.check())

    def test_unowned_cv2_module_is_rejected(self):
        self.record["files"].remove("lib/python3.11/site-packages/cv2/__init__.py")
        self.assertFalse(self.check())

    def test_non_headless_or_mismatched_conda_provider_is_rejected(self):
        self.record["build"] = "qt6_test_0"
        self.assertFalse(self.check())
        self.record["build"] = "headless_test_0"
        self.record["version"] = "4.12.0"
        self.assertFalse(self.check())


if __name__ == "__main__":
    unittest.main()
