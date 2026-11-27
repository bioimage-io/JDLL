"""Native Cellpose 3.1.1.1 training with JDLL progress events for 2D grayscale data.

From the Cellpose 3.1.1.1 source root, run git apply with the path to
native-training-events.patch and install that patched build in the worker.
Put this directory on the worker's Python import path. For a Windows Appose
service, preload NumPy, torch, cellpose.models, and this module in service.init().
The host supplies the Appose task, CellposeModel, and lists of YX images and
integer instance masks. Each run saves its weights and previews in a new directory.
"""

import json
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from PIL import Image
from cellpose import train, transforms, utils
from jdll_unet.metrics import instance_pq_statistics


def publish_preview(model, image, truth, epoch, directory):
    # Use the training-set diameter saved in these weights, as on a fresh reload.
    prediction, _, _ = model.eval(
        image, channels=[0, 0], diameter=float(model.net.diam_labels.item())
    )
    matched_iou, n_true, n_pred = instance_pq_statistics(prediction, truth)
    n_objects = n_true + n_pred
    pq = float(2 * matched_iou / n_objects) if n_objects else 1.0

    gray = (np.clip(transforms.normalize99(image.astype(np.float32)), 0, 1) * 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    folder = directory / f"epoch_{epoch:04d}"
    folder.mkdir()
    image_path = folder / "image.png"
    Image.fromarray(rgb).save(image_path)
    item = {"image_path": str(image_path)}
    for name, labels in (("target", truth), ("prediction", prediction)):
        overlay = rgb.copy()
        overlay[utils.masks_to_outlines(labels)] = (255, 160, 0)
        path = folder / f"{name}.png"
        Image.fromarray(overlay).save(path)
        item[f"{name}_path"] = str(path)

    manifest = folder / "preview.json"
    payload = json.dumps({"epoch": epoch, "items": [item]})
    manifest.write_text(payload, encoding="utf-8")
    latest = directory / "latest.json"
    temporary = directory / "latest.tmp"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(latest)
    return {"preview_path": str(manifest), "latest_preview_path": str(latest)}, pq


def train_with_updates(task, model, train_images, train_masks, val_images, val_masks,
                       output_dir, *, epochs, report_every=10, **training_options):
    """Return the native weights path. training_options are passed to train_seg."""
    if report_every < 1 or epochs < 1:
        raise ValueError("epochs and report_every must be positive")
    if len(val_images) == 0 or len(val_images) != len(val_masks):
        raise ValueError("Supply paired validation images and instance masks")
    # Keep the original field for comparison across epochs, before native preprocessing.
    image = np.array(val_images[0], copy=True)
    truth = np.array(val_masks[0], copy=True)
    if image.ndim != 2 or image.shape != truth.shape:
        raise ValueError("This example expects a 2D grayscale field and a matching instance mask")
    parent = Path(output_dir).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix="cellpose-", dir=parent))

    def send(kind, message, epoch, step, total_steps, **details):
        info = dict(type=kind, epoch=epoch, step=step, total_epochs=epochs,
                    total_steps=total_steps, **details)
        task.update(message=message, current=step, maximum=total_steps, info=info)

    def on_step(*, epoch, step, total_steps, loss, learning_rate):
        if step == 1 or step % report_every == 0 or step == total_steps:
            send("progress", f"Cellpose training step {step}/{total_steps}",
                 epoch, step, total_steps, losses={"train/total_loss": loss},
                 learning_rate=learning_rate)

    def on_validation_end(*, epoch, step, total_steps, train_loss, val_loss):
        losses = {"train/total_loss": train_loss, "val/total_loss": val_loss}
        send("validation", f"Cellpose validation completed for epoch {epoch}",
             epoch, step, total_steps, status="completed", losses=losses)
        try:
            paths, pq = publish_preview(model, image, truth, epoch, run_dir)
        except Exception as error:
            # An optional preview failure must not discard the native checkpoint.
            send("warning", f"Cellpose preview failed: {error}", epoch, step, total_steps)
            metrics = {}
        else:
            metrics = {"preview/panoptic_quality": pq}
            send("preview", f"Cellpose held-out field, epoch {epoch}",
                 epoch, step, total_steps, **paths)
        send("progress", f"Cellpose epoch {epoch} results", epoch, step, total_steps,
             losses=losses, metrics=metrics)

    path, _, _ = train.train_seg(
        model.net, train_data=train_images, train_labels=train_masks,
        test_data=val_images, test_labels=val_masks,
        channels=[0, 0], normalize=True, save_path=str(run_dir), n_epochs=epochs,
        on_step=on_step, on_validation_end=on_validation_end, validate_each_epoch=True,
        **training_options,
    )
    return str(path)
