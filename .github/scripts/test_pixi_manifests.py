"""Dependency and rendering contracts, runnable without installing model packages."""

import unittest

from render_jdll_pixi import TOMLS, render, tomllib


class PixiManifestTests(unittest.TestCase):
    def test_all_platform_templates_render_and_keep_backend_revisions_in_sync(self):
        pins = {}
        templates = [p for p in TOMLS.glob("jdll-pixi-*.toml") if p.name != "jdll-pixi-stardist.toml"]
        self.assertEqual(len(templates), 8)
        for template in templates:
            data = tomllib.loads(render(template, "jdll-ci"))
            features = data["feature"]
            self.assertEqual(data["environments"]["default"], ["runtime", "pytorch", "core"])
            self.assertEqual(set(data["environments"]), {"default"})
            self.assertEqual(features["runtime"]["dependencies"]["python"], "3.11.*")
            core = features["core"]["pypi-dependencies"]
            self.assertIn("ultralytics-opencv-headless", core)
            self.assertNotIn("opencv-python", core)
            self.assertEqual(core["numpy"], "==2.0.2")
            self.assertEqual(core["zarr"], "==2.18.7")
            self.assertEqual(features["runtime"]["pypi-dependencies"]["tifffile"], "==2025.5.10")
            for package in ("jdll-unet", "jdll-denoise", "jdll-cross-goose"):
                dependency = core[package]
                self.assertEqual(len(dependency["rev"]), 40)
                self.assertEqual(pins.setdefault(package, dependency["rev"]), dependency["rev"])
            self.assertNotIn("cross-goose", features)
            self.assertIn("cellpose", core)
            self.assertIn("lightning", core)
            all_direct = {name for feature in features.values() for name in feature.get("pypi-dependencies", {})}
            self.assertFalse(all_direct & {"biapy", "careamics", "timm", "torch-fidelity", "pytorch-msssim", "tensorflow"})

    def test_cuda_indices_are_explicit_and_do_not_share_cpu_wheels(self):
        for platform in ("lin", "win"):
            for cuda in ("11.8", "12.1", "12.4"):
                data = tomllib.loads(render(TOMLS / f"jdll-pixi-{platform}-x86-cuda.toml", "cuda-test", cuda))
                for package in ("torch", "torchvision"):
                    dependency = data["feature"]["pytorch"]["pypi-dependencies"][package]
                    self.assertTrue(dependency["index"].endswith("cu" + cuda.replace(".", "")))
            data = tomllib.loads(render(TOMLS / f"jdll-pixi-{platform}-x86-no-cuda.toml", "cpu-test"))
            self.assertTrue(data["feature"]["pytorch"]["pypi-dependencies"]["torch"]["index"].endswith("/cpu"))

    def test_all_intel_backends_use_newer_conda_builds_in_the_same_environment(self):
        for suffix, minimum, version, opencv in (("", "12.0", "==2.14.1", "==4.13.0"),
                                                 ("-big-sur", "11.0", "==2.12.0", "==4.13.0"),
                                                 ("-catalina", "10.15", "==2.7.1", "==4.11.0")):
            data = tomllib.loads(render(TOMLS / f"jdll-pixi-mac-x86{suffix}.toml", "intel-test"))
            self.assertEqual(data["environments"], {"default": ["runtime", "pytorch", "core"]})
            self.assertEqual(data["system-requirements"]["macos"], minimum)
            self.assertIn("Intel macOS", data["workspace"]["description"])
            dependencies = data["feature"]["pytorch"]["dependencies"]
            self.assertEqual(dependencies["pytorch"]["version"], version)
            self.assertIn("torchvision", dependencies)
            self.assertEqual(dependencies["numpy"], "==2.0.2")
            self.assertEqual(dependencies["py-opencv"], {"version": opencv, "build": "headless*"})
            self.assertEqual(data["feature"]["core"]["pypi-dependencies"]["opencv-python-headless"], opencv)

    def test_stardist_is_isolated_and_keeps_keras_two(self):
        data = tomllib.loads(render(TOMLS / "jdll-pixi-stardist.toml", "stardist-test"))
        self.assertIn("stardist", data["pypi-dependencies"])
        self.assertNotIn("torch", data["pypi-dependencies"])
        for feature in data["feature"].values():
            dependencies = feature.get("pypi-dependencies", {})
            self.assertNotIn("tensorflow-macos", dependencies)
            if "tensorflow" in dependencies:
                self.assertIn(dependencies["tensorflow"], {"==2.10.1", "==2.15.1"})
        for platform in ("linux-x86-64-no-cuda", "win-x86-64-no-cuda"):
            self.assertEqual(data["feature"][platform]["pypi-dependencies"], {"tensorflow-cpu": "==2.15.1"})
        self.assertIn("%CONDA_PREFIX%", render(TOMLS / "jdll-pixi-stardist.toml", "stardist-test"))

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

    def test_opencv_uses_real_headless_packages_without_local_compatibility_projects(self):
        self.assertFalse((TOMLS.parent / "python/opencv-python-compat/pyproject.toml").exists())
        data = tomllib.loads(render(TOMLS / "jdll-pixi-mac-arm.toml", "arm-test"))
        self.assertEqual(data["system-requirements"]["macos"], "12.0")
        self.assertEqual(data["feature"]["core"]["pypi-dependencies"]["opencv-python-headless"], "==4.10.0.84")


if __name__ == "__main__":
    unittest.main()
