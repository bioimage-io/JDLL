"""Diameter policy without downloading weights or importing a Cellpose runtime."""
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HELPER = Path(__file__).parents[2] / "main/resources/python/cellpose_inference.py"
spec = importlib.util.spec_from_file_location("cellpose_inference", HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class DiameterTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.events = []
        self.task = SimpleNamespace(update=lambda **event: self.events.append(event))
        self.model = SimpleNamespace(cp=SimpleNamespace(diam_mean=30, diam_labels=19))
        self.created = []
        self.calls = []
        self.estimate = [23]
        test = self

        class SizeModel:
            def __init__(self, pretrained_size, cp_model):
                test.created.append(Path(pretrained_size).name)

            def eval(self, images, channels, channel_axis):
                test.calls.append((images, channels, channel_axis))
                return test.estimate, None

        module = ModuleType("cellpose.models")
        module.SizeModel = SizeModel
        self.modules = patch.dict(sys.modules, {"cellpose.models": module})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def resolve(self, weights="custom", requested=None):
        return helper.resolve_cellpose_diameter(self.model, [np.zeros((4, 5, 3))],
                                               [[1, 2]], requested, str(self.root / weights), self.task)

    def test_official_model_prefers_its_size_file_and_caches_estimator(self):
        for name in ("size_aaa.npy", "size_cyto3.npy"):
            (self.root / name).touch()
        self.assertEqual(23, self.resolve("cyto3"))
        self.assertEqual(23, self.resolve("cyto3"))
        self.assertEqual(["size_cyto3.npy"], self.created)
        self.assertEqual(-1, self.calls[0][2])
        self.assertFalse(self.events[0]["info"]["warning"])

    def test_custom_selection_is_sorted_and_ambiguity_is_reported(self):
        for name in ("size_z.npy", "size_a.npy"):
            (self.root / name).touch()
        self.assertEqual(23, self.resolve())
        self.assertEqual(["size_a.npy"], self.created)
        self.assertTrue(self.events[0]["message"].startswith("Warning:"))
        self.assertEqual("estimated", self.events[0]["info"]["source"])

    def test_no_estimator_uses_model_diameter_and_warns(self):
        self.assertEqual(19, self.resolve())
        self.assertEqual("model_default", self.events[0]["info"]["source"])
        self.assertTrue(self.events[0]["info"]["warning"])

    def test_official_model_does_not_use_unrelated_size_weights(self):
        (self.root / "size_nucleitorch_0.npy").touch()
        self.assertEqual(19, self.resolve("cyto3"))
        self.assertFalse(self.created)
        self.assertIn("No matching size model", self.events[-1]["message"])

    def test_invalid_estimate_falls_back_explicitly(self):
        (self.root / "size_custom.npy").touch()
        for estimate in ([np.nan], [0], [-1], [1, 2]):
            self.estimate = estimate
            self.assertEqual(19, self.resolve())
            self.assertIn("estimation failed", self.events[-1]["message"])

    def test_manual_diameter_skips_estimation(self):
        (self.root / "size_custom.npy").touch()
        self.assertEqual(42, self.resolve(requested=42))
        self.assertFalse(self.created)
        self.assertEqual("manual", self.events[-1]["info"]["source"])

    def test_rejects_invalid_diameter(self):
        for value in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                self.resolve(requested=value)

    def test_helper_survives_appose_task_namespace(self):
        exports = {}
        task = SimpleNamespace(export=lambda **values: exports.update(values))
        binding = {"task": task}
        exec(compile(HELPER.read_text(), str(HELPER), "exec"), binding, binding)
        task.export(resolve_cellpose_diameter=binding["resolve_cellpose_diameter"])
        fresh_task = dict(exports, model=self.model, task=self.task, path=str(self.root / "custom"))
        exec("result = resolve_cellpose_diameter(model, [], [], 25, path, task)", fresh_task, fresh_task)
        self.assertEqual(25, fresh_task["result"])


if __name__ == "__main__":
    unittest.main()
