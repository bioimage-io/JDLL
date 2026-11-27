"""Appose API for training and fine-tuning native Cellpose instance models."""
import errno
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import tifffile
import torch
from cellpose import models, train
from jdll_unet.metrics import instance_pq_statistics

from jdll_cellpose_data import prepare_dataset
from jdll_devices import torch_device
from jdll_validation_control import FullValidationSchedule
from jdll_training_continuation import TrainingRun

BACKEND = "cellpose3-cpnet"


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def disk_full(error):
    return (error.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", -1))
            or getattr(error, "winerror", None) in (39, 112))


def save_weights(net, path):
    temporary = path.with_suffix(".tmp")
    net.save_model(str(temporary))
    os.replace(temporary, path)


def disk_budget(train_fields, validation_fields, epochs, checkpoint_bytes):
    """Estimate peak staged TIFFs, concurrent checkpoints, metadata and optional previews."""
    staged = 0
    preview_pixels = []
    for training, fields in ((True, train_fields), (False, validation_fields)):
        for pair in fields:
            shape = pair.domain_shape
            h, w = shape[-2:]
            planes = shape[0] if len(shape) == 3 else 1
            if training and not pair.region and pair.plane_positive_counts:
                planes = sum(count > 0 for count in pair.plane_positive_counts)
            # Two float32 image channels and four float32 target channels. Allow for
            # TIFF headers, filesystem allocation and per-plane manifest/log entries.
            staged += planes * (24 * h * w + 16384 + 4 * (len(str(pair.image)) + len(str(pair.mask))))
            if not training:
                stride = max(1, math.ceil(max(h, w) / 768))
                pixels = math.ceil(h / stride) * math.ceil(w / stride)
                preview_pixels = sorted(preview_pixels + [pixels] * min(3, planes), reverse=True)[:3]
    # Best, last, the native trainer's periodic copy, and one atomic replacement.
    # Include serialization overhead, size weights, the epoch history and both logs.
    # Up to three live bundles during atomic best/last rotation, with two Adam moments each.
    # Reserve space for native weight snapshots on volumes without hard-link support.
    checkpoint_reserve = 16 * (checkpoint_bytes + 1024**2) + 1024**2 + 8192 * epochs
    pixels = sum(preview_pixels)
    preview_epoch = 4 * pixels + 32768
    preview_total = 12 * pixels + 32768 + epochs * preview_epoch
    return staged + checkpoint_reserve, preview_total, checkpoint_reserve


def initialize_cpnet(config, device):
    # Use the same full-precision native CPnet for both starting points.
    model = models.CellposeModel(pretrained_model=False, device=device, mkldnn=False, diam_mean=30.)
    origin = {"kind": "scratch"}
    if config.get("base_weights"):
        path = Path(config["base_weights"])
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Earlier Cellpose 3-compatible checkpoints predate the two diameter buffers.
        # All learned parameters must still match CPnet exactly.
        reference = 17. if path.name == "nucleitorch_0" or config.get("base_model") == "nuclei" else 30.
        state.setdefault("diam_mean", torch.tensor([reference]))
        state.setdefault("diam_labels", state["diam_mean"].clone())
        model.net.load_state_dict(state, strict=True)
        del state
        model.diam_mean = float(model.net.diam_mean.item())
        if not math.isfinite(model.diam_mean) or model.diam_mean <= 0:
            raise ValueError(f"Invalid reference diameter in {path}")
        origin = dict(kind="fine_tune", base=config.get("base_model", path.name),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return model, origin


def evaluate(model, fields, diameter, channels, check, progress, collect_preview=False):
    counts = np.zeros(3, dtype=np.float64)
    items = []
    for index, field in enumerate(fields):
        check()
        image = tifffile.imread(field["image"])
        truth = tifffile.imread(field["flow"])[0].astype(np.int32)
        # The staged fields have already been channel-mapped and percentile-normalized.
        prediction, _, _ = model.eval(np.moveaxis(image, 0, -1), channels=channels,
                                      channel_axis=-1, normalize=False, diameter=diameter)
        counts += instance_pq_statistics(prediction, truth)
        if collect_preview and index < 3:
            stride = max(1, math.ceil(max(truth.shape) / 768))
            assets = {name: data[..., ::stride, ::stride].copy()
                      for name, data in (("image", image), ("target", truth), ("prediction", prediction))}
            items.append(dict(source_path=field["source"], plane=field["plane"], stride=stride, assets=assets))
        progress(index + 1, len(fields))
    overlap, truth_count, pred_count = counts
    return float(2 * overlap / (truth_count + pred_count)) if truth_count + pred_count else 1., items


def save_preview(folder, epoch, items):
    shared = folder.parent / "fields"
    folder.mkdir(parents=True, exist_ok=True)
    shared.mkdir(exist_ok=True)
    for index, item in enumerate(items):
        for name, data in item["assets"].items():
            path = (folder if name == "prediction" else shared) / f"{index}-{name}.npy"
            if not path.exists():
                try:
                    with path.open("wb") as stream:
                        np.save(stream, data, allow_pickle=False)
                except OSError:
                    path.unlink(missing_ok=True)
                    raise
            item["assets"][name] = {"path": str(path)}
    manifest = {"epoch": epoch, "dimensions": "2d", "items": items}
    write_json(folder / "preview.json", manifest)
    write_json(folder.parent / "latest.json", manifest)


class Cancelled(Exception):
    pass


def train_model(config, output_dir, task, sources=None, request_dir=None):
    """Run in an exclusively reserved directory. The returned weights open in stock Cellpose."""
    out = Path(output_dir).resolve()
    backend = config.get("backend", BACKEND)
    if backend != BACKEND:
        raise ValueError(f"Unsupported Cellpose training backend: {backend}")
    additional = int(config["epochs"])
    run = TrainingRun(out, additional, config.get("base_weights"), config.get("_jdll_legacy"))
    run.info["dataset"] = str(config["dataset"])
    epochs = run.total
    channels = list(config.get("channels", [0, 0]))
    if additional < 1 or epochs < 2:
        raise ValueError("Cellpose needs at least two epochs. The first warm-up epoch has a zero learning rate.")
    if len(channels) != 2 or any(type(c) is not int or c < 0 for c in channels):
        raise ValueError("Expected two nonnegative channel indices")
    control = Path(request_dir) if request_dir else out / "requests"
    control.mkdir(exist_ok=True)
    history = []
    best = (-math.inf, -math.inf)
    execution = {}
    def notify(message):
        with (out / "training.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        task.update(message=message)
    def check():
        if task.cancel_requested:
            raise Cancelled("Cellpose training cancelled")
    schedule = FullValidationSchedule(request_path=control, update=task.update)
    def status(state, **fields):
        write_json(out / "run.json", dict(status=state, completed_epochs=run.start + len(history), **execution, **fields))
        run.finish(state)
    status("running")
    try:
        write_json(out / "config.json", {k: v for k, v in config.items() if k not in ("base_weights", "_jdll_legacy")})
        torch.manual_seed(42)
        device = torch.device(torch_device(config["device"], task.update))
        model, origin = initialize_cpnet(config, device)
        saved = None
        if run.selected:
            selected = run.selected
            if selected["restore"]:
                saved = torch.load(run.parent / selected["state"], map_location="cpu", weights_only=True)
                model.net.load_state_dict(saved["weights"])
            else:
                model.net.load_state_dict(torch.load(run.parent / selected["weights"], map_location="cpu", weights_only=True))
            model.diam_mean = float(model.net.diam_mean.item())
            origin = dict(kind="continuation", base=run.parent.name, epoch=run.start)
            notify(f"Continuing from {selected['kind']} at epoch {run.start}. " +
                   ("Restoring its optimizer." if saved else "Optimizer state unavailable; initializing a fresh optimizer."))

        with tempfile.TemporaryDirectory(prefix=".cellpose-data-", dir=out) as tmp:
            work = Path(tmp)
            previews = True
            checkpoint_reserve = 0
            def disable_previews():
                nonlocal previews
                previews = False
                message = "Validation preview unavailable: insufficient disk space. Training continues."
                task.update(message=message, info={"type": "preview_unavailable", "reason": message})
            def check_space(train_pairs, val_pairs):
                nonlocal checkpoint_reserve
                size = sum(t.numel() * t.element_size() for t in model.net.state_dict().values())
                required, optional, checkpoint_reserve = disk_budget(train_pairs, val_pairs, epochs, size)
                free = shutil.disk_usage(out).free
                if free < required:
                    raise OSError(errno.ENOSPC,
                                  f"Cellpose training requires an estimated {required / 1e9:.2f} GB for this "
                                  f"dataset. Only {free / 1e9:.2f} GB is free at {out}.")
                if free < required + optional:
                    disable_previews()
            data = prepare_dataset(config["dataset"], work, channels, notify, check, sources, check_space, run)
            write_json(out / "dataset.json", {s: [{k: v for k, v in f.items() if k not in ("image", "flow")}
                                                    for f in fields] for s, fields in data.items()})
            training, validation = data["train"], data["val"]
            regular = [validation[i] for i in np.linspace(0, len(validation)-1, min(8, len(validation)), dtype=int)]
            if not any(field["instances"] for field in regular):
                regular[-1] = next(field for field in validation if field["instances"])
            batch = min(4 if device.type == "cuda" else 2, len(training))
            execution.update(device=str(device), batch_size=batch)
            status("running")
            report_every = max(1, math.ceil(len(training) / batch / 100))
            native_channels = [1, 2 if channels[1] else 0]
            optimizer = None
            future_rates = None
            batches = None

            def checkpoint(epoch, score, improved, last=True):
                state = out / f"training-epoch-{epoch}.pt"
                tmp = state.with_suffix(".tmp")
                moments = optimizer.state_dict()
                for group in moments["param_groups"]:
                    group["lr"] = float(group["lr"])
                torch.save(dict(weights=model.net.state_dict(), optimizer=moments,
                                rates=future_rates.tolist(), batches=int(batches)), tmp)
                os.replace(tmp, state)
                for kind in (("best",) if improved else ()) + (("last",) if last else ()):
                    weights = out / (kind + ".pt")
                    save_weights(model.net, weights)
                    if kind == "best":
                        # The model picker must be able to open the first published checkpoint after a hard exit.
                        write_json(out / "model.json", dict(format_version=1, backend=backend, weights="best.pt",
                                   channels=channels, origin=origin, best_epoch=epoch, reference_diameter_px=model.diam_mean,
                                   training_diameter_px=float(model.net.diam_labels.item())))
                    run.checkpoint(kind, epoch, weights, state, list(score))
                live = {slot["state"] for slot in run.info["checkpoints"].values()}
                for obsolete in out.glob("training-epoch-*.pt"):
                    if obsolete.name not in live:
                        obsolete.unlink()

            def on_train_begin(native_optimizer, rates, steps_per_epoch):
                nonlocal optimizer, future_rates, batches, best, saved
                optimizer, future_rates, batches = native_optimizer, rates, steps_per_epoch
                if not run.selected:
                    return
                if saved:
                    optimizer.load_state_dict(saved["optimizer"])
                    if epochs == run.previous["epochs"] and batches == saved["batches"]:
                        rates[run.start:epochs] = saved["rates"][run.start:epochs]
                    else:
                        lr = optimizer.param_groups[0]["lr"]
                        # A zero-rate warm-up checkpoint still needs to complete warm-up.
                        if lr > 0:
                            rates[run.start:epochs] = np.linspace(lr, min(lr, rates[epochs-1]), additional)
                        notify(f"Replanned the remaining native learning-rate schedule for {additional} epochs.")
                    saved = None
                previous_score = run.selected.get("score")
                if run.previous.get("validation") != run.info["validation"] or previous_score is None:
                    pq, _ = evaluate(model, regular, float(model.net.diam_labels.item()), native_channels,
                                     check, lambda n, total: None)
                    # A baseline has no native crop loss. Require a strictly better PQ to displace it.
                    best = (pq, 0.)
                    notify(f"Validation changed. Scored the starting weights before training: baseline PQ {pq:.4f}.")
                else:
                    best = tuple(previous_score)
                checkpoint(run.start, best, True)

            def send(event_type, epoch, step, total_steps, **details):
                task.update(current=step, maximum=total_steps, info=dict(
                    type=event_type, epoch=epoch, step=step, total_epochs=epochs, total_steps=total_steps, **details))
            def on_step(*, epoch, step, total_steps, loss, learning_rate):
                check()
                if not math.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {step} on {device}. Last validated weights are preserved.")
                if step == 1 or step == total_steps or step % report_every == 0:
                    send("progress", epoch, step, total_steps, losses={"train/total_loss": loss},
                         learning_rate=learning_rate)
            def on_validation_end(*, epoch, step, total_steps, train_loss, val_loss):
                nonlocal best
                check()
                if not all(map(math.isfinite, (train_loss, val_loss))):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch} on {device}")
                diameter = float(model.net.diam_labels.item())
                def val_progress(n, maximum):
                    send("validation", epoch, step, total_steps, status="progress", current=n, maximum=maximum, unit="fields")
                folder = out / "previews" / f"epoch-{epoch:04d}"
                send("validation", epoch, step, total_steps, status="started")
                pq, items = evaluate(model, regular, diameter, native_channels, check, val_progress, previews)
                losses = {"train/total_loss": train_loss, "val/total_loss": val_loss}
                metrics = {"val/panoptic_quality": pq}
                score = (pq, -val_loss)
                improved = score > best
                checkpoint(epoch, score, improved)
                send("checkpoint", epoch, step, total_steps, kind="last", path=str(out / "last.pt"))
                if improved:
                    best = score
                    send("checkpoint", epoch, step, total_steps, kind="best", path=str(out / "best.pt"))
                row = dict(epoch=epoch, train_loss=train_loss, validation_loss=val_loss, panoptic_quality=pq)
                history.append(row)
                write_json(out / "metrics.json", history)
                status("running")
                if previews:
                    existing = (out / "best.pt", out / "last.pt", work / "models" / "final")
                    remaining = checkpoint_reserve - sum(p.stat().st_size for p in existing if p.is_file())
                    preview_bytes = sum(a.nbytes for item in items for a in item["assets"].values()) + 65536
                    if shutil.disk_usage(out).free < remaining + preview_bytes:
                        disable_previews()
                    else:
                        try:
                            save_preview(folder, epoch, items)
                        except OSError as error:
                            if not disk_full(error) and shutil.disk_usage(out).free >= preview_bytes:
                                raise
                            disable_previews()
                        else:
                            send("preview", epoch, step, total_steps, preview_path=str(folder / "preview.json"),
                                 latest_preview_path=str(folder.parent / "latest.json"))
                send("validation", epoch, step, total_steps, status="completed", metrics=metrics)
                send("progress", epoch, step, total_steps, losses=losses, metrics=metrics)
                notify(f"Epoch {epoch}/{epochs}  train loss {train_loss:.5f}  validation loss {val_loss:.5f}  PQ {pq:.4f}")
                if schedule.consume(epoch):
                    schedule.started(epoch)
                    def full_progress(n, maximum):
                        schedule.emit("progress", epoch, current=n, maximum=maximum, unit="fields")
                    try:
                        full_pq = pq if len(regular) == len(validation) else evaluate(
                            model, validation, diameter, native_channels, check, full_progress)[0]
                    except Cancelled:
                        raise
                    except Exception as error:
                        row["full_validation_error"] = str(error)
                        schedule.finish(epoch, "failed", message=f"Full validation failed: {error}")
                    else:
                        row["full_panoptic_quality"] = full_pq
                        schedule.finish(epoch, "completed", message=f"Full validation PQ: {full_pq:.4f}",
                                        metrics={"panoptic_quality": full_pq})
                    write_json(out / "metrics.json", history)
            schedule.emit("ready", supported=True)
            notify(f"Training native CPnet on {len(training)} planes; validating on {len(validation)}. Device: {device}.")
            train.train_seg(model.net, train_files=[f["image"] for f in training],
                            train_labels_files=[f["flow"] for f in training],
                            test_files=[f["image"] for f in validation], test_labels_files=[f["flow"] for f in validation],
                            load_files=False, normalize=False, min_train_masks=0, n_epochs=epochs,
                            batch_size=batch, save_path=work, model_name="final",
                            on_step=on_step, on_validation_end=on_validation_end, validate_each_epoch=True,
                            on_train_begin=on_train_begin, start_epoch=run.start)
            check()
            if len(history) != additional or not (out / "best.pt").is_file():
                raise RuntimeError("Native training finished without all expected validation checkpoints")
            notify("Fitting the diameter estimator for the best validated checkpoint.")
            # Regression extracts styles on larger 512-pixel crops and no longer needs the segmentation optimizer.
            optimizer = None
            model.net.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=True))
            params = train.train_size(model.net, str(work / "best.pt"), train_files=[f["image"] for f in training],
                                      train_labels_files=[f["flow"] for f in training], load_files=False,
                                      normalize=False, min_train_masks=0, batch_size=1 if device.type == "cpu" else batch)
            if not all(np.isfinite(params[key]).all() for key in ("A", "smean", "diam_mean", "ymean")):
                raise ValueError("The fitted diameter estimator contains non-finite coefficients.")
            check()
            os.replace(work / "best.pt_size.npy", out / "best.pt_size.npy")
        status("complete")
        return str(out / "best.pt")
    except Cancelled:
        status("cancelled")
        task.cancel()
        return None
    except Exception as exc:
        reason = exc.strerror if isinstance(exc, OSError) and disk_full(exc) else str(exc)
        message = f"Training failed: {reason}"
        task.update(message=message, info={"type": "training_error"})
        try:
            with (out / "training.log").open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")
            status("failed", error=reason)
        except OSError as log_error:
            if not disk_full(log_error):
                raise
        raise
    finally:
        schedule.close()
