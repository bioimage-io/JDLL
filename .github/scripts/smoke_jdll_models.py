"""Small real model operations; no pretrained weights or external datasets required."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import traceback


def finite(array, shape=None):
    import numpy as np

    value = np.asarray(array)
    if shape is not None and value.shape != shape:
        raise AssertionError(f"Expected {shape}, got {value.shape}")
    if not np.isfinite(value).all():
        raise AssertionError("Non-finite model output")


def image():
    import numpy as np

    y, x = np.mgrid[:64, :64]
    labels = np.zeros((64, 64), np.uint16)
    labels[(y - 22) ** 2 + (x - 22) ** 2 < 8**2] = 1
    labels[(y - 43) ** 2 + (x - 43) ** 2 < 7**2] = 7
    raw = (labels > 0).astype(np.float32) + np.random.default_rng(1).uniform(0, 0.05, labels.shape)
    return raw.astype(np.float32), labels


def unet(device, root):
    import numpy as np
    import tifffile
    import zarr
    import torch
    from jdll_unet.appose_api import infer, train
    from jdll_unet.config import ArchitectureConfig
    from jdll_unet.model import build_unet

    raw, labels = image()
    compressed = root / "region.tif"
    tifffile.imwrite(compressed, raw, tile=(16, 16), compression="zlib")
    with tifffile.TiffFile(compressed) as tiff:
        store = tiff.aszarr()
        try:
            region = zarr.open(store, mode="r")[8:24, 9:31]
            np.testing.assert_array_equal(region, raw[8:24, 9:31])
        finally:
            store.close()
    dataset = root / "dataset"
    for split, count in (("train", 2), ("val", 1)):
        for kind in ("images", "masks"):
            (dataset / split / kind).mkdir(parents=True)
        for index in range(count):
            tifffile.imwrite(dataset / split / "images" / f"{index}.tif", raw)
            tifffile.imwrite(dataset / split / "masks" / f"{index}.tif", labels)
    result = train({
        "dataset_path": str(dataset), "output_dir": str(root / "unet"), "model_name": "smoke",
        "architecture": "resenc-tiny-2d", "task": "instance_friendly", "device": device,
        "epochs": 1, "steps_per_epoch": 1, "patch_size": [32, 32],
        "batch_size": 1, "effective_batch_size": 1, "num_workers": 0,
        "preview_count": 0, "validation": {"mode": "light", "light_steps": 1},
    })
    prediction = infer({"model_path": result["model_path"], "device": device, "object_size": 15},
                       {"image": raw[None]})["outputs"]
    finite(prediction["labels"], raw.shape)
    for key in ("foreground_probability", "boundary_probability", "distance_probability"):
        finite(prediction[key], raw.shape)
    # Exercise volumetric kernels/decoder and the 2.5D channel stack without a large dataset.
    for dimensions, shape in (("2.5d", (1, 3, 32, 32)), ("3d", (1, 1, 8, 32, 32))):
        config = ArchitectureConfig(name=f"resenc-tiny-{dimensions}", dimensions=dimensions,
                                    input_channels=shape[1], output_channels=3, base_channels=4, depth=2)
        model = build_unet(config).to(device)
        tensor = torch.rand(shape, device=device)
        output = model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        finite(output.detach().cpu().numpy(), (shape[0], 3, *shape[2:]))
        output.float().square().mean().backward()


def denoise(device, root):
    import numpy as np
    from jdll_denoise import denoise as run

    raw, _ = image()
    for method in ("noise2fast", "zs_n2n", "structn2v", "bm3d_bm4d"):
        config = {"method": method, "axes": "yx", "device": "cpu" if method == "bm3d_bm4d" else device,
                  "effort": "quick", "seed": 1}
        if method != "bm3d_bm4d":
            config["method_config"] = {"steps": 1}
        if method == "structn2v":
            config["method_config"].update(patch_size=[32, 32], batch_size=1)
        result = run(raw, config)
        finite(result["denoised"], raw.shape)
    # Load and invoke the native BM4D binary too, not just the Python package.
    result = run(np.stack([raw] * 8), {"method": "bm3d_bm4d", "axes": "zyx",
                                    "device": "cpu", "effort": "quick"})
    finite(result["denoised"], (8, *raw.shape))


def cellpose(device, root):
    import numpy as np
    import torch
    from cellpose.denoise import CellposeDenoiseModel, DenoiseModel

    # Same restoration + segmentation wrapper used by JDLL, with random weights
    # instead of downloading either pretrained network during the environment test.
    model = CellposeDenoiseModel(pretrained_model=False, model_type=None, restore_type=None,
                                device=torch.device(device))
    # DenoiseModel.eval expects a checkpoint filename even for a synthetic network.
    checkpoint = root / "restoration.pt"
    model.dn.net.save_model(str(checkpoint))
    model.dn = DenoiseModel(pretrained_model=str(checkpoint), device=torch.device(device))
    raw, _ = image()
    labels, flows, _, restored = model.eval(np.stack([raw, raw], axis=-1), channels=[1, 2], channel_axis=-1,
                                 diameter=30, niter=5, bsize=64, batch_size=1, augment=False,
                                 flow_threshold=0, cellprob_threshold=0)
    finite(labels, raw.shape)
    finite(flows[1])
    finite(restored)


def yolo(device, root):
    from types import SimpleNamespace
    import numpy as np
    import torch
    from ultralytics import YOLO
    from torchvision.ops import nms

    # Ultralytics performs NMS on CPU for MPS inputs; test that same fallback.
    nms_device = "cpu" if device == "mps" else device
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=torch.float32, device=nms_device)
    scores = torch.tensor([0.9, 0.8, 0.7], device=nms_device)
    if nms(boxes, scores, 0.5).cpu().tolist() != [0, 2]:
        raise AssertionError("Torchvision's native NMS produced the wrong result")
    model = YOLO("yolov8n.yaml")
    raw, _ = image()
    rgb = np.repeat((raw * 200).clip(0, 255).astype(np.uint8)[..., None], 3, axis=-1)
    result = model.predict(rgb, imgsz=64, device=device, verbose=False)[0]
    finite(result.boxes.data.detach().cpu().numpy())
    model.model.to(device).train()
    model.model.requires_grad_(True)
    # The Trainer normally turns the architecture's argument dict into a namespace.
    if isinstance(model.model.args, dict):
        model.model.args = SimpleNamespace(**model.model.args)
    batch = {"img": torch.rand(2, 3, 64, 64, device=device),
             "batch_idx": torch.tensor([0, 1], device=device),
             "cls": torch.zeros(2, 1, device=device),
             "bboxes": torch.full((2, 4), 0.4, device=device)}
    loss, _ = model.model.loss(batch)
    finite(loss.detach().cpu().numpy())
    loss.sum().backward()


def cross_goose(device, root):
    import numpy as np
    import torch
    import yaml
    from jdll_cross_goose.appose_api import infer
    from jdll_cross_goose.checkpoints import default_config
    from jdll_cross_goose.model import create_model

    config = default_config()
    config["model"]["n_steps"] = 3
    config["adapter"] = {"object_size_statistics": {"median": 20.0}}
    folder = root / "cross-goose"
    folder.mkdir()
    (folder / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    model = create_model(config)
    torch.save({"state_dict": model.state_dict()}, folder / "weights.ckpt")
    raw, _ = image()
    result = infer({"model_path": str(folder), "device": device, "tile_size": 64, "tile_overlap": 16},
                   {"image": np.stack([raw, raw])})
    finite(result["outputs"]["labels"], raw.shape)
    finite(result["outputs"]["cell_probability"], raw.shape)
    model = model.to(device).train()
    output = model.image_to_maps(torch.rand(2, 2, 32, 32, device=device))
    loss = sum(value.float().square().mean() for value in output.values())
    finite(loss.detach().cpu().numpy())
    loss.backward()


def stardist(device, root):
    import numpy as np
    import tensorflow as tf
    from stardist.models import Config2D, Config3D, StarDist2D, StarDist3D

    gpus = tf.config.list_physical_devices("GPU")
    if device != "cpu" and not gpus:
        raise RuntimeError(f"Requested {device}, but TensorFlow reports no GPU")
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    for dimensions in (2, 3):
        shape = (32, 32) if dimensions == 2 else (8, 16, 16)
        config_class, model_class = (Config2D, StarDist2D) if dimensions == 2 else (Config3D, StarDist3D)
        kwargs = {"n_rays": 8, "n_channel_in": 1, "grid": (1,) * dimensions, "use_gpu": False,
                  "train_patch_size": shape}
        kwargs.update(unet_n_depth=1, unet_n_filter_base=4)
        model = model_class(config_class(**kwargs), name=None, basedir=None)
        raw = np.random.default_rng(1).random(shape).astype(np.float32)
        probability, distances = model.predict(raw, axes="YX" if dimensions == 2 else "ZYX")
        finite(probability)
        finite(distances)
        labels, _ = model.predict_instances(raw, prob_thresh=0.1, nms_thresh=0.4,
                                            axes="YX" if dimensions == 2 else "ZYX")
        finite(labels, shape)
        # Use the real TensorFlow model to check gradients as well as inference.
        with tf.GradientTape() as tape:
            outputs = model.keras_model(raw[None, ..., None], training=True)
            loss = sum(tf.reduce_mean(tf.square(value)) for value in outputs)
        gradients = tape.gradient(loss, model.keras_model.trainable_variables)
        if not gradients or any(value is None or not np.isfinite(value.numpy()).all() for value in gradients):
            raise AssertionError("Missing or non-finite StarDist gradients")


CHECKS = {"unet": unet, "denoise": denoise, "cellpose": cellpose, "yolo": yolo,
          "cross-goose": cross_goose, "stardist": stardist}


def check_dependencies():
    from packaging.requirements import Requirement

    installed = {}
    errors = []
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        installed[name] = distribution.version
        for text in distribution.requires or ():
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                version = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                errors.append(f"{name} requires missing {requirement}")
            else:
                if requirement.specifier and version not in requirement.specifier:
                    errors.append(f"{name} requires {requirement}; found {version}")
    if "opencv-python" in installed and "opencv-python-headless" in installed:
        alias = metadata.distribution("opencv-python")
        requirements = alias.requires or []
        # The only allowed alias contains metadata, never a second cv2 implementation.
        metadata_only = all(".dist-info/" in str(path).replace("\\", "/") for path in alias.files or ())
        if (alias.version != "4.11.0.86+jdll.headless" or not alias.files or not metadata_only
                or requirements != ["opencv-python-headless==4.11.0.86"]):
            errors.append("Both OpenCV implementations are installed and can overwrite the same cv2 module")
    if errors:
        raise AssertionError("Unsatisfied installed dependencies:\n" + "\n".join(errors))
    return installed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=CHECKS, default=["unet", "denoise", "cellpose", "yolo", "cross-goose"])
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    (args.report.parent / "yolo-config").mkdir(exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.report.resolve().parent / "yolo-config"))
    versions = {}
    for name in ("appose", "torch", "torchvision", "numpy", "jdll-unet", "jdll-denoise", "jdll-cross-goose",
                 "cellpose", "ultralytics-opencv-headless", "tensorflow", "tensorflow-cpu", "tensorflow-metal", "stardist"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    report = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
              "device": args.device, "versions": versions, "models": {}}
    if "torch" in versions:
        import torch
        torch.set_num_threads(2)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but no CUDA GPU is available")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS, but no Metal device is available")
    try:
        report["installed_packages"] = check_dependencies()
        for name in args.models:
            started = time.perf_counter()
            try:
                with tempfile.TemporaryDirectory(prefix="jdll-smoke-") as temporary:
                    CHECKS[name](args.device, Path(temporary))
            except Exception:
                report["models"][name] = {"status": "failed", "traceback": traceback.format_exc()}
            else:
                report["models"][name] = {"status": "passed"}
            report["models"][name]["seconds"] = time.perf_counter() - started
            print(name, report["models"][name], flush=True)
    except Exception:
        report["environment_error"] = traceback.format_exc()
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if any(value["status"] != "passed" for value in report["models"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
