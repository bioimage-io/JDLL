"""Persist the trainable network as well as the EMA that Ultralytics uses for validation."""
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch


def copy_weights(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, target)


def continuing_trainer(base, run, notify):
    class Trainer(base):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if run.selected and weights is not None and dict(weights.names) != dict(self.data["names"]):
                best = run.previous["checkpoints"].get("best")
                if best and run.selected["kind"] != "best" and (run.parent / best["weights"]).is_file():
                    checkpoint = torch.load(run.parent / best["weights"], map_location="cpu", weights_only=False)
                    weights = checkpoint.get("ema") or checkpoint["model"]
            return super().get_model(cfg=cfg, weights=weights, verbose=verbose)

        def _setup_train(self):
            self.segment = None
            slot = run.selected
            saved = None
            if slot:
                source = torch.load(run.parent / slot["weights"], map_location="cpu", weights_only=False)
                names = (source.get("ema") or source["model"]).names
                if dict(names) != dict(self.data["names"]):
                    run.use_fine_tuning()
                    self.epochs = self.args.epochs = run.total
                    slot = None
                    notify(message="Detection classes changed. Transferring compatible weights with a fresh optimizer.")
                elif slot["restore"]:
                    saved = torch.load(run.parent / slot["state"], map_location="cpu", weights_only=True)
                    self.args.optimizer = saved["optimizer_name"]
                    for name in ("cos_lr", "lrf", "warmup_epochs", "warmup_bias_lr", "warmup_momentum"):
                        if name in source["train_args"]:
                            setattr(self.args, name, source["train_args"][name])
                del source
            super()._setup_train()
            if not slot:
                return
            self.start_epoch = run.start
            self.epoch = run.start - 1
            if saved:
                self.model.load_state_dict(saved["model"])
                self.ema.ema.load_state_dict(saved["ema"])
                self.ema.updates = saved["updates"]
                self.optimizer.load_state_dict(saved["optimizer"])
                if self.amp:
                    self.scaler.load_state_dict(saved["scaler"])
                self.scheduler.load_state_dict(saved["scheduler"])
                same = run.total == run.previous["epochs"] and len(self.train_loader) == saved["batches"]
                self.segment = saved.get("segment") if same else {
                    "start": run.start, "end": run.total,
                    "initial": [g["lr"] for g in self.optimizer.param_groups],
                    "minimum": [lr * self.args.lrf for lr in saved["original_lrs"]]}
                self.original_lrs = saved["original_lrs"]
                if self.segment:
                    self.args.warmup_epochs = 0
                    segment = self.segment
                    def factor(epoch):
                        t = min(1., max(0., (epoch-segment["start"]) / max(1, segment["end"]-segment["start"]-1)))
                        return .5 * (1 + math.cos(math.pi*t)) if self.args.cos_lr else 1-t
                    self.scheduler.base_lrs = list(segment["initial"])
                    self.scheduler.lr_lambdas = [
                        (lambda epoch, floor=min(1., end/max(start, 1e-30)): floor+(1-floor)*factor(epoch))
                        for start, end in zip(segment["initial"], segment["minimum"])]
                notify(message=f"Restored the {slot['kind']} optimizer at epoch {run.start}. " +
                       ("Keeping its LR schedule." if same else "Replanned LR over the remaining training."))
            else:
                notify(message="Resuming from weights with a fresh optimizer.")
            self.scheduler.last_epoch = self.start_epoch - 1
            self.fitness = slot.get("score")
            if run.validation_changed or self.fitness is None:
                self.loss_items = torch.zeros(len(self.loss_names), device=self.device)
                self.loss = torch.tensor(0., device=self.device)
                self.metrics, self.fitness = self.validate()
                notify(message=f"Scored starting weights on current validation: fitness {self.fitness:.5f}.")
            self.best_fitness = self.fitness
            copy_weights(run.parent / slot["weights"], self.best)
            self.publish(self.best, True)

        def publish(self, native, best):
            directory = run.directory / "training-state"
            directory.mkdir(exist_ok=True)
            epoch = self.epoch + 1
            state = directory / f"epoch-{epoch}.pt"
            tmp = state.with_suffix(".tmp")
            # Warm-up uses numpy.interp, so optimizer groups and scheduler rates can contain NumPy scalars.
            def portable(value):
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, dict):
                    return {k: portable(v) for k, v in value.items()}
                if isinstance(value, (tuple, list)):
                    return type(value)(portable(v) for v in value)
                return value
            torch.save(dict(model=self.model.state_dict(), ema=self.ema.ema.state_dict(), updates=self.ema.updates,
                            optimizer=portable(self.optimizer.state_dict()), optimizer_name=type(self.optimizer).__name__,
                            scaler=self.scaler.state_dict(), scheduler=portable(self.scheduler.state_dict()),
                            segment=portable(self.segment), original_lrs=portable(getattr(self, "original_lrs", self.scheduler.base_lrs)),
                            batches=len(self.train_loader)), tmp)
            os.replace(tmp, state)
            for kind in (("last", "best") if best else ("last",)):
                path = (run.directory / (run.directory.name + ".pt") if kind == "best" else directory / "last.pt")
                copy_weights(native, path)
                run.checkpoint(kind, epoch, path, state, float(self.fitness))
            live = {str(run.directory / item["state"]) for item in run.info["checkpoints"].values()}
            for path in directory.glob("epoch-*.pt"):
                if str(path) not in live:
                    path.unlink()

        def save_model(self):
            saved = super().save_model()
            if saved:
                self.publish(self.last, self.best_fitness == self.fitness)
            return saved
    return Trainer
