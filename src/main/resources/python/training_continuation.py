"""On-disk contract shared by the DeepIcy training workers and model picker."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from dataclasses import replace


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def source_pairs(root):
    return read_json(Path(root) / "jdll-source-pairs.json")


def lock_worker(stream):
    stream.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.lockf(stream, fcntl.LOCK_EX | fcntl.LOCK_NB, 1)


def active_worker(folder):
    path = folder / ".training.lock"
    if not path.exists():
        return False
    try:
        with path.open("r+b") as stream:
            lock_worker(stream)
    except OSError:
        return True
    return False


def select_checkpoint(directory, info):
    slots = info.get("checkpoints", {})
    for field in ("state", "weights"):
        for kind in ("best", "last"):
            slot = slots.get(kind, {})
            if not slot.get("weights") or not (directory / slot["weights"]).is_file():
                continue
            snapshot = slot.get("snapshot")
            if snapshot and not (directory / snapshot).is_file():
                if field == "state":
                    continue
                snapshot = None
            if slot.get(field) and (directory / slot[field]).is_file():
                if field == "state" and slot.get("state_bytes") is not None and (directory / slot[field]).stat().st_size != slot["state_bytes"]:
                    continue
                if field == "state" and any(not (directory / name).is_file() for name in slot.get("companions", [])):
                    continue
                return dict(slot, weights=snapshot or slot["weights"], kind=kind, restore=field == "state")
    return None


class TrainingRun:
    def __init__(self, directory, epochs, base=None, legacy=None):
        self.directory = Path(directory).absolute()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "continuation.json"
        if self.path.exists():
            raise FileExistsError(f"Training directory already contains a run: {self.directory}")
        self.parent = None
        self.selected = None
        self.previous = {}
        if base and Path(base).exists():
            base = Path(base).absolute()
            candidates = [base] if base.is_dir() else [base.parent, base.parent.parent]
            if self.directory == candidates[0] or (candidates[0].name in {"weights", "checkpoints", "training-state"}
                                                  and self.directory == candidates[0].parent):
                raise ValueError("Select a new output directory for training from this model")
            for folder in candidates:
                previous = read_json(folder / "continuation.json")
                if not previous and legacy and Path(legacy["directory"]) == folder:
                    previous = legacy
                if previous and previous["status"] != "complete" and not active_worker(folder):
                    slot = select_checkpoint(folder, previous)
                    if slot:
                        if folder == self.directory:
                            raise ValueError("A continuation requires a new output directory")
                        self.parent, self.previous, self.selected = folder, previous, slot
                    break
        self.start = int(self.selected["epoch"]) if self.selected else 0
        self.total = self.start + epochs
        self.worker = (self.directory / ".training.lock").open("a+b")
        lock_worker(self.worker)
        if self.path.exists():
            self.worker.close()
            raise FileExistsError(f"Training directory already contains a run: {self.directory}")
        self.info = dict(version=1, status="running", epochs=self.total, checkpoints={})
        if self.selected:
            self.info["parent"] = str(self.parent)
        self.flush()

    def flush(self):
        atomic_json(self.path, self.info)

    @property
    def validation_changed(self):
        return not self.info.get("validation") or self.info["validation"] != self.previous.get("validation")

    def use_fine_tuning(self):
        self.total -= self.start
        self.start = 0
        self.selected = None
        self.info.update(epochs=self.total)
        self.flush()

    def finish(self, status):
        self.info["status"] = status
        self.flush()
        if status != "running":
            self.worker.close()

    def checkpoint(self, kind, epoch, weights, state=None, score=None):
        weights = Path(weights).absolute()
        slot = dict(epoch=int(epoch), weights=weights.relative_to(self.directory).as_posix())
        if state is not None:
            state = Path(state).absolute()
            if state == weights:
                # Native U-Net and Lightning replace the public filenames each epoch.
                # Keep the previous generation valid until its index has been replaced.
                generation = state.with_name(f"continuation-epoch-{epoch}{state.suffix}")
                if not generation.exists():
                    try:
                        os.link(state, generation)
                    except OSError:
                        tmp = generation.with_suffix(".tmp")
                        shutil.copyfile(state, tmp)
                        os.replace(tmp, generation)
                state = generation
            else:
                # The public best/last filenames can advance before a new index is committed.
                # Tie the optimizer to native weights from exactly this epoch.
                snapshot = self.directory / ".continuation" / f"weights-{epoch}{weights.suffix}"
                snapshot.parent.mkdir(exist_ok=True)
                if not snapshot.exists():
                    # Keras can overwrite its public best HDF5 file in place.
                    if not (weights.suffix == ".h5" and kind == "best"):
                        try:
                            os.link(weights, snapshot)
                        except OSError:
                            pass
                    if not snapshot.exists():
                        temporary = snapshot.with_suffix(".tmp")
                        shutil.copyfile(weights, temporary)
                        os.replace(temporary, snapshot)
                slot["snapshot"] = snapshot.relative_to(self.directory).as_posix()
            slot["state"] = state.relative_to(self.directory).as_posix()
            slot["state_bytes"] = state.stat().st_size
            if state.suffix == ".index":
                slot["companions"] = [p.relative_to(self.directory).as_posix()
                                      for p in state.parent.glob(state.stem + ".data-*")]
                schedule = state.with_suffix(".json")
                if schedule.is_file():
                    slot["companions"].append(schedule.relative_to(self.directory).as_posix())
        if score is not None:
            slot["score"] = score
        previous = self.info["checkpoints"].get(kind, {})
        old = previous.get("state")
        self.info["checkpoints"][kind] = slot
        self.flush()
        if old and Path(old).name.startswith("continuation-epoch-") and not any(
                s.get("state") == old for s in self.info["checkpoints"].values()):
            (self.directory / old).unlink(missing_ok=True)
        snapshot = previous.get("snapshot")
        if snapshot and not any(s.get("snapshot") == snapshot for s in self.info["checkpoints"].values()):
            (self.directory / snapshot).unlink(missing_ok=True)

    @staticmethod
    def pair_lists(root, pair_type):
        sources = source_pairs(root)
        if sources:
            return [[pair_type(Path(i), Path(m), Path(i).stem) for i,m in sources[group]]
                    for group in ("train","val")]
        return None

    def source_names(self):
        split = read_json(self.directory / "split.json")
        names = {}
        for pair in split.get("train", []) + split.get("val", []):
            image = self.source_root / pair[0]
            if image.is_file():
                stat = image.stat()
                names[stat.st_dev, stat.st_ino] = pair
        return names

    @staticmethod
    def domain_mask(pair):
        if hasattr(pair, "source"):
            mask = pair.source.raw(True)
            return mask[tuple(slice(a, b) for a, b in pair.region)] if pair.region else mask
        from jdll_unet.geometry import load_domain_mask
        return load_domain_mask(pair, dimensions=None, raw=True)

    @staticmethod
    def region_pair(pair, bounds):
        if hasattr(pair, "plane_positive_counts"):
            # Native dataset planning uses per-region plane counts and eligible Z centers.
            from jdll_unet.geometry import with_region
            return with_region(pair, tuple(bounds), "spatial_holdout")
        return replace(pair, region=tuple(bounds))

    def split_annotated_domain(self, pair, train, val):
        import numpy as np
        mask = self.domain_mask(pair)
        for axis in sorted(range(mask.ndim), key=lambda i: -mask.shape[i]):
            view = np.moveaxis(mask, axis, 0)
            minimum = getattr(self, "minimum_region_shape", (2,) * mask.ndim)[axis]
            axes = tuple(range(1, view.ndim))
            cut_objects = np.any((view[:-1] != 0) & (view[:-1] == view[1:]), axis=axes)
            count = np.cumsum(np.count_nonzero(view, axis=axes))
            cuts = [int(i) for i in np.flatnonzero(~cut_objects) + 1
                    if 0 < count[i-1] < count[-1] and min(i, len(view)-i) >= minimum]
            if cuts:
                left = list(pair.region or [(0, n) for n in mask.shape])
                right = left.copy()
                start, end = left[axis]
                cut = start + min(cuts, key=lambda i: abs(i/len(view) - .8))
                left[axis], right[axis] = (start, cut), (cut, end)
                return ([p for p in train if p is not pair] + [self.region_pair(pair, left)],
                        val + [self.region_pair(pair, right)])
        raise ValueError(f"No disjoint holdout with complete annotated objects on both sides of {pair.image}. "
                         "Supply another annotated image.")

    def restore_domains(self, train, val, notify):
        """Recover disjoint regions from a previous automatic single-source holdout."""
        if getattr(self, "spatial_source", None):
            pair = next(p for p in train if os.path.samefile(p.image, self.spatial_source))
            return self.split_annotated_domain(pair, train, val)
        if not self.parent or not (self.directory / "split.json").exists():
            return train, val
        regions = read_json(self.parent / "split.json").get("spatial", [])
        if not regions:
            return train, val
        names = self.source_names()
        result = {"train": [], "val": []}
        seen = set()
        for group, pairs in (("train", train), ("val", val)):
            for pair in pairs:
                stat = Path(pair.image).stat()
                name = names.get((stat.st_dev, stat.st_ino))
                saved = [r for r in regions if r["source"] == name]
                if not saved:
                    result[group].append(pair)
                    continue
                if tuple(name) in seen:
                    continue
                seen.add(tuple(name))
                if any(len(r["region"]) != len(pair.spatial_shape) or any(
                       not 0 <= a < b <= n for (a,b),n in zip(r["region"], pair.spatial_shape)) for r in saved):
                    result["train"].append(pair)
                    notify(f"Source dimensions changed for {name[0]}. Rebuilding its validation holdout.")
                    continue
                for r in saved:
                    result[r["split"]].append(self.region_pair(pair, [tuple(v) for v in r["region"]]))
        train, val = result["train"], result["val"]
        if val and not any(self.domain_mask(p).any() for p in val):
            annotated = (p for p in sorted(train, key=lambda p: bool(p.region)) if self.domain_mask(p).any())
            candidate = next(annotated, None)
            if candidate is None:
                raise ValueError("No annotated objects remain in the training or validation sources.")
            notify(f"The previous validation regions contain no objects. Bringing annotations from {Path(candidate.image).name} into validation.")
            if next(annotated, None) is None:
                return self.split_annotated_domain(candidate, train, val)
            train = [p for p in train if p is not candidate]
            val.append(candidate)
        return train, val

    def record_domains(self, train, val):
        if not (self.directory / "split.json").is_file():
            if getattr(self, "spatial_source", None):
                digest = hashlib.sha256(self.info.get("validation", "").encode())
                for image, mask, region in val:
                    if region and os.path.samefile(image, self.spatial_source):
                        digest.update(validation_signature(Path(image).parent, [(image,mask,region)]).encode())
                self.info["validation"] = digest.hexdigest()
                self.flush()
            return
        names = self.source_names()
        result = {"train": [], "val": [], "spatial": []}
        held = []
        for group, pairs in (("train", train), ("val", val)):
            for image, mask, region in pairs:
                stat = Path(image).stat()
                name = names[stat.st_dev, stat.st_ino]
                if name not in result[group]:
                    result[group].append(name)
                if region:
                    result["spatial"].append(dict(source=name, region=region, split=group))
                if group == "val":
                    held.append((self.source_root / name[0], self.source_root / name[1], region))
        atomic_json(self.directory / "split.json", result)
        self.info["validation"] = validation_signature(self.source_root, held)
        self.flush()


def validation_signature(root, fields):
    """Fingerprint original validation pixels/annotations once, including spatial holdouts."""
    root = Path(root).absolute()
    digest = hashlib.sha256()
    files = {}
    for image, annotation, region in sorted(fields, key=lambda f: tuple(str(v) for v in f)):
        for name in (image, annotation):
            path = Path(name).absolute()
            if path not in files:
                file_hash = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        file_hash.update(chunk)
                files[path] = file_hash.hexdigest()
            digest.update(json.dumps([os.path.relpath(path, root).replace(os.sep, "/"), files[path]]).encode())
        digest.update(json.dumps(region).encode())
    return digest.hexdigest()


def train_dense(trainer, config, **options):
    config = dict(config)
    run = TrainingRun(config.get("model_dir") or config["output_dir"], int(config["epochs"]),
                      config.get("base_model"), config.pop("_jdll_legacy", None))
    run.source_root = Path(config.pop("_jdll_source_root", config["dataset_path"])).absolute()
    run.spatial_source = config.pop("_jdll_spatial_source", None)
    run.info["dataset"] = str(run.source_root)
    signature = run.directory / "validation.sha256"
    if signature.is_file():
        run.info["validation"] = signature.read_text()
    run.flush()
    try:
        result = trainer(dict(config, _jdll_run=run), **options)
    except BaseException as error:
        run.finish("cancelled" if "cancel" in type(error).__name__.lower() else "failed")
        raise
    run.finish("cancelled" if result.get("cancelled") else "complete")
    return result
