"""Run with python scripts/test_training_continuation.py. No DL environment is required."""
from pathlib import Path
import os
import importlib.util
import sys
import subprocess
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/main/resources/python"))
from training_continuation import TrainingRun, atomic_json, select_checkpoint, validation_signature


class ContinuationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_fallback_order_and_source_immutability(self):
        parent = TrainingRun(self.root / "parent", 100)
        for kind, epoch in (("best", 38), ("last", 50)):
            weights = parent.directory / (kind + ".pt")
            state = parent.directory / (kind + ".state")
            weights.write_bytes(b"weights")
            state.write_bytes(b"optimizer")
            parent.checkpoint(kind, epoch, weights, state, .5)
        parent.finish("cancelled")
        unchanged = parent.path.read_bytes()
        best = parent.directory / "best.pt"
        best.rename(parent.directory / "moved.pt")
        self.assertEqual(select_checkpoint(parent.directory, parent.info)["kind"], "last")
        (parent.directory / "moved.pt").rename(best)
        for kind, epoch, restore in (("best", 38, True), ("last", 50, True), ("best", 38, False)):
            child = TrainingRun(self.root / f"child-{kind}-{restore}", 7, parent.directory)
            self.assertEqual((child.start, child.total, child.selected["kind"], child.selected["restore"]),
                             (epoch, epoch + 7, kind, restore))
            child.finish("complete")
            if restore:
                (parent.directory / (kind + ".state")).unlink()
        self.assertEqual(parent.path.read_bytes(), unchanged)
        fresh = TrainingRun(self.root / "fine", 3, child.directory)
        self.assertIsNone(fresh.selected)
        fresh.finish("complete")

    def test_native_snapshot_survives_public_weight_replacement(self):
        run = TrainingRun(self.root / "run", 4)
        path = run.directory / "last.pt"
        path.write_bytes(b"one")
        run.checkpoint("last", 1, path, path)
        slot = select_checkpoint(run.directory, run.info)
        replacement = run.directory / "next.tmp"
        replacement.write_bytes(b"two")
        os.replace(replacement, path)
        self.assertEqual((run.directory / slot["state"]).read_bytes(), b"one")
        run.checkpoint("last", 2, path, path)
        self.assertFalse((run.directory / slot["state"]).exists())
        run.finish("complete")

    def test_separate_state_keeps_matching_native_weights_across_interrupted_publish(self):
        for suffix in (".pt", ".h5"):
            run = TrainingRun(self.root / suffix[1:], 3)
            public = run.directory / ("best" + suffix)
            state = run.directory / "epoch-1.state"
            public.write_bytes(b"first weights")
            state.write_bytes(b"first optimizer")
            run.checkpoint("best", 1, public, state, .6)
            if suffix == ".h5":
                public.write_bytes(b"second weights")
            else:
                temporary = public.with_suffix(".tmp")
                temporary.write_bytes(b"second weights")
                os.replace(temporary, public)
            # The process stopped after the native writer ran and before the new index.
            slot = select_checkpoint(run.directory, run.info)
            self.assertEqual((run.directory / slot["weights"]).read_bytes(), b"first weights")
            self.assertEqual((run.directory / slot["state"]).read_bytes(), b"first optimizer")
            run.finish("cancelled")

    def test_abrupt_worker_exit_releases_lock_and_keeps_last_committed_epoch(self):
        parent = self.root / "parent"
        program = """
import os, sys
from training_continuation import TrainingRun
run = TrainingRun(sys.argv[1], 100)
weights, state = run.directory / "best.pt", run.directory / "epoch-38.state"
weights.write_bytes(b"epoch 38")
state.write_bytes(b"optimizer 38")
run.checkpoint("best", 38, weights, state, .8)
next_weights = run.directory / "next.tmp"
next_weights.write_bytes(b"epoch 39")
os.replace(next_weights, weights)
os._exit(19)
"""
        env = dict(os.environ, PYTHONPATH=sys.path[0], PYTHONDONTWRITEBYTECODE="1")
        process = subprocess.run([sys.executable, "-c", program, str(parent)], env=env, check=False)
        self.assertEqual(process.returncode, 19)
        child = TrainingRun(self.root / "child", 62, parent)
        self.addCleanup(child.finish, "complete")
        self.assertEqual((child.start, child.total), (38, 100))
        self.assertEqual((parent / child.selected["weights"]).read_bytes(), b"epoch 38")
        self.assertEqual((parent / child.selected["state"]).read_bytes(), b"optimizer 38")

    @unittest.skipUnless(importlib.util.find_spec("cellpose"), "requires the Cellpose training environment")
    def test_cellpose_repairs_an_edited_spatial_validation_mask(self):
        import numpy as np
        import tifffile
        import training_continuation
        sys.modules.setdefault("jdll_training_continuation", training_continuation)
        from cellpose_data import source_splits
        from jdll_unet.geometry import DomainReader
        from jdll_unet.image_reading import image_reading_session
        data = self.root / "data"
        for kind in ("images", "masks"):
            (data / kind).mkdir(parents=True)
        def field(name, cells):
            array = np.zeros((64, 64), np.uint16)
            for i, (y, x) in enumerate(cells, start=1):
                array[y:y+4, x:x+4] = i
            for kind in ("images", "masks"):
                tifffile.imwrite(data / kind / name, array)
        def split(run):
            with image_reading_session() as session:
                return source_splits(data, DomainReader(session=session), lambda _: None, lambda: None, None, run)
        parent = TrainingRun(self.root / "parent", 3)
        field("old.tif", [(2, 3), (54, 52)])
        _, val = split(parent)
        held = val[0].region
        weights = parent.directory / "best.pt"
        weights.touch()
        parent.checkpoint("best", 1, weights)
        parent.finish("cancelled")
        field("old.tif", [(2, 3)])
        field("new.tif", [(16, 16)])
        child = TrainingRun(self.root / "child", 2, parent.directory)
        self.addCleanup(child.finish, "complete")
        train, val = split(child)
        self.assertEqual([p.image.name for p in train], ["old.tif"])
        self.assertIn(("old.tif", held), [(p.image.name, p.region) for p in val])
        self.assertIn("new.tif", [p.image.name for p in val])

    def test_validation_relocation_edits_and_regions(self):
        def fingerprint(directory, region):
            return validation_signature(directory, [(directory / "image", directory / "mask", region)])
        dirs = [self.root / name for name in ("a", "b")]
        for directory in dirs:
            directory.mkdir()
            (directory / "image").write_bytes(b"i")
            (directory / "mask").write_bytes(b"m")
        original = fingerprint(dirs[0], [[0, 10]])
        self.assertEqual(original, fingerprint(dirs[1], [[0, 10]]))
        self.assertNotEqual(original, fingerprint(dirs[1], [[0, 9]]))
        (dirs[1] / "mask").write_bytes(b"n")
        self.assertNotEqual(original, fingerprint(dirs[1], [[0, 10]]))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires NumPy for pixel geometry")
    def test_stardist_single_image_holdout_has_annotations_and_fits_the_patch(self):
        from types import SimpleNamespace
        import numpy as np
        from stardist_data import DatasetStore
        mask = np.zeros((128, 128), np.int32)
        mask[30:34, 30:34], mask[65:70, 65:70] = 1, 2
        store = SimpleNamespace(channels=1, update=lambda **_: None)
        source = SimpleNamespace(shape=mask.shape, raw=lambda _: mask, store=store, image="field", mask="mask")
        labels = [SimpleNamespace(source=source)]
        _, training, _, held = DatasetStore.spatial_split(store, [], labels, (32, 32), .2)
        for group in (training, held):
            self.assertTrue(group[0].source.raw(True).any())
            self.assertGreaterEqual(min(group[0].source.shape), 32)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires NumPy for pixel geometry")
    def test_only_annotated_source_is_split_without_cutting_an_object(self):
        import numpy as np
        from dataclasses import dataclass
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[3:8, 3:8] = 1
        mask[24:30, 24:30] = 2

        class Source:
            def raw(self, labels):
                return mask

        @dataclass
        class Pair:
            image: Path
            source: Source
            region: tuple = ()
        image = self.root / "annotated"
        image.touch()
        pair = Pair(image, Source())
        run = TrainingRun(self.root / "run", 1)
        run.spatial_source = image
        train, val = run.restore_domains([pair], [], lambda _: None)
        def objects(p):
            return set(np.unique(mask[tuple(slice(a, b) for a,b in p.region)])) - {0}
        self.assertTrue(objects(train[0]))
        self.assertTrue(objects(val[0]))
        self.assertTrue(objects(train[0]).isdisjoint(objects(val[0])))
        run.finish("complete")

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires NumPy for pixel geometry")
    def test_previously_annotated_validation_region_is_now_background(self):
        from dataclasses import dataclass
        import numpy as np

        class Source:
            def raw(self, labels):
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[2:6, 2:6] = 1
                return mask

        @dataclass
        class Pair:
            image: Path
            spatial_shape = (32, 32)
            source = Source()
            region: tuple = ()

        pairs = [Pair(self.root / name) for name in ("old", "new")]
        for pair in pairs:
            pair.image.touch()
        names = [[p.image.name, p.image.name + "-mask"] for p in pairs]
        parent = self.root / "parent"
        parent.mkdir()
        split = dict(train=names, val=[names[0]], spatial=[
            dict(source=names[0], split="train", region=[[0, 16], [0, 32]]),
            dict(source=names[0], split="val", region=[[16, 32], [0, 32]])])
        atomic_json(parent / "split.json", split)
        run = TrainingRun(self.root / "run", 2)
        run.parent, run.source_root = parent, self.root
        atomic_json(run.directory / "split.json", split)
        messages = []
        train, val = run.restore_domains(pairs, [], messages.append)
        self.assertEqual([p.image.name for p in train], ["old"])
        self.assertEqual([p.image.name for p in val], ["old", "new"])
        self.assertEqual(val[0].region, ((16, 32), (0, 32)))
        self.assertTrue(messages)
        run.finish("complete")


if __name__ == "__main__":
    unittest.main()
