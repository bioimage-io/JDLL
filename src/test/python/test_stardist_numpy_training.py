"""Opt-in one-epoch smoke run of the generated StarDist task on mapped volumes."""

import ast
import hashlib
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.dont_write_bytecode = True
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
code = ast.parse(Path(sys.argv[1]).read_text())

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    output = root / "model"
    for split in ("train", "val"):
        images, masks = root / "dataset" / split / "images", root / "dataset" / split / "masks"
        images.mkdir(parents=True)
        masks.mkdir()
        rng = np.random.default_rng(42)
        image = rng.random((8, 1, 128, 160), dtype=np.float32) * 1000
        labels = np.zeros((8, 128, 160), np.uint32)
        labels[1:6, 25:50, 40:70] = 70000
        labels[2:7, 70:95, 90:120] = 300
        np.save(images / "case.npy", image)
        np.save(masks / "case_mask.npy", labels)
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "dataset").rglob("*.npy")}
    overrides = {"n_dim": 3, "n_rays": 8, "grid": [1, 2, 2], "train_patch_size": [4, 32, 32],
                 "train_batch_size": 1, "train_epochs": 1, "train_steps_per_epoch": 1,
                 "unet_n_depth": 1, "unet_n_filter_base": 4, "validation_preview_count": 0,
                 "data_loading": {"cache_mb": 0}, "validation": {"preview_count": 0}}
    body = []
    for node in code.body:
        name = node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None
        if name in ("data_dir", "output_dir"):
            value = str(root / "dataset") if name == "data_dir" else str(output)
            node.value = ast.Constant(value) if name == "data_dir" else ast.Call(ast.Name("Path", ast.Load()), [ast.Constant(value)], [])
        body.append(node)
        if name == "config":
            body.extend(ast.parse("config.update(" + repr(overrides) + ")").body)
    code.body = body
    ast.fix_missing_locations(code)
    events = []

    class Task:
        outputs = {}

        def update(self, **event):
            events.append(event)

        def export(self, **values):
            scope.update(values)

    task = Task()
    scope = {"task": task}
    exec(compile(code, "<generated StarDist training>", "exec"), scope)
    assert task.outputs["result"] == str(output)
    assert (output / "weights_best.h5").is_file()
    assert (output / "weights_last.h5").is_file()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    print("Generated StarDist training completed one epoch on read-only NumPy volumes")
