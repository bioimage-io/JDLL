"""Manifest rendering checks, runnable without installing model packages."""

import unittest

from render_jdll_pixi import TOMLS, render, tomllib


class PixiManifestTests(unittest.TestCase):
    def test_all_platform_templates_render(self):
        templates = list(TOMLS.glob("jdll-pixi-*.toml"))
        self.assertTrue(templates)
        for template in templates:
            with self.subTest(template=template.name):
                data = tomllib.loads(render(template, "jdll-ci"))
                self.assertEqual(data["workspace"]["name"], "jdll-ci")
                self.assertTrue(data["workspace"]["platforms"])

    def test_cuda_indices_are_explicit_and_do_not_share_cpu_wheels(self):
        for platform in ("lin", "win"):
            for cuda in ("11.8", "12.1", "12.4"):
                data = tomllib.loads(render(TOMLS / f"jdll-pixi-{platform}-x86-cuda.toml", "cuda-test", cuda))
                for package in ("torch", "torchvision"):
                    dependency = data["feature"]["pytorch"]["pypi-dependencies"][package]
                    self.assertTrue(dependency["index"].endswith("cu" + cuda.replace(".", "")))
            data = tomllib.loads(render(TOMLS / f"jdll-pixi-{platform}-x86-no-cuda.toml", "cpu-test"))
            self.assertTrue(data["feature"]["pytorch"]["pypi-dependencies"]["torch"]["index"].endswith("/cpu"))

    def test_unknown_cuda_is_rejected(self):
        with self.assertRaises(ValueError):
            render(TOMLS / "jdll-pixi-lin-x86-cuda.toml", "invalid", "12.7")

    def test_model_specific_environments_are_rejected(self):
        with self.assertRaises(ValueError):
            render(TOMLS / "jdll-pixi-lin-x86-no-cuda.toml", "invalid", environment="cross-goose")
        with self.assertRaises(ValueError):
            render(TOMLS / "jdll-pixi-mac-arm.toml", "invalid", environment="macos-catalina")

    def test_old_intel_cli_profiles_resolve_to_explicit_versioned_templates(self):
        for suffix in ("big-sur", "catalina"):
            self.assertEqual(render(TOMLS / "jdll-pixi-mac-x86.toml", "intel-test", environment=f"macos-{suffix}"),
                             render(TOMLS / f"jdll-pixi-mac-x86-{suffix}.toml", "intel-test"))


if __name__ == "__main__":
    unittest.main()
