"""Execute JDLL's generated API call with distinct primary and auxiliary predictions."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

CODE = Path(sys.argv.pop(1)).read_text(encoding="utf-8")


class InferenceOutputTest(unittest.TestCase):
    def run_bridge(self, task, outputs):
        image = object()
        request = {}
        events = []

        def infer(config, inputs, callback):
            request.update(config)
            self.assertIs(image, inputs["image"])
            callback({"message": "Done", "current": 1, "maximum": 1})
            return {"outputs": outputs}

        namespace = {"_jdll_unet_task": task, "_jdll_unet_device": "cpu", "input_image": image,
                     "jdll_unet_infer": infer, "task": SimpleNamespace(update=lambda **event: events.append(event))}
        exec(compile(CODE, "generated_unet_inference", "exec"), namespace)
        self.assertEqual("model.pt", request["model_path"])
        self.assertEqual("Done", events[-1]["message"])
        return request, namespace["_jdll_unet_labels"]

    def test_binary_ignores_connected_component_labels_even_if_returned(self):
        mask = [[0, 1, 0], [1, 0, 1]]
        request, result = self.run_bridge("binary_semantic", {"mask": mask, "labels": [[0, 7, 0], [8, 0, 9]]})
        self.assertIs(mask, result)
        self.assertEqual({0, 1}, {value for row in result for value in row})
        self.assertNotIn("postprocessing", request)

    def test_semantic_keeps_class_ids_instead_of_auxiliary_instance_ids(self):
        mask = [[0, 4, 9], [9, 4, 0]]
        request, result = self.run_bridge("multiclass_semantic", {"mask": mask, "labels": [[0, 1, 2], [3, 4, 0]]})
        self.assertIs(mask, result)
        self.assertNotIn("postprocessing", request)

    def test_instance_keeps_object_ids_instead_of_a_binary_mask(self):
        labels = [[0, 17, 70000], [0, 17, 70000]]
        request, result = self.run_bridge("instance_friendly", {"mask": [[0, 1, 1], [0, 1, 1]], "labels": labels})
        self.assertIs(labels, result)
        self.assertNotIn("postprocessing", request)


if __name__ == "__main__":
    unittest.main()
