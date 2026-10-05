"""Small CPU integration tests against the actual StarDist/TensorFlow backend."""

import importlib.util
import json
import os
import random
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'

import numpy as np
from stardist import Rays_GoldenSpiral
from stardist.models import Config2D, Config3D, StarDist2D, StarDist3D

resource = Path(__file__).resolve().parents[2] / 'main/resources/python/stardist_validation.py'
sys.path.insert(0, str(resource.parent))
spec = importlib.util.spec_from_file_location('jdll_stardist_validation', resource)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
from stardist_data import DatasetStore, median_object_extents
from stardist_sampling import build_2d_plan, build_plan


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.events = []

    def model(self, dimensions=3, batch=1):
        config = dict(n_channel_in=1, grid=(1,) * dimensions, unet_n_depth=1,
                      unet_n_filter_base=2, net_conv_after_unet=2,
                      train_patch_size=(4, 16, 16) if dimensions == 3 else (16, 16),
                      train_batch_size=batch, train_reduce_lr=None, train_tensorboard=False,
                      train_foreground_only=0, train_sample_cache=False)
        if dimensions == 3:
            config = Config3D(rays=Rays_GoldenSpiral(8), unet_pool=(1, 2, 2), **config)
            model = StarDist3D(config, name='model', basedir=str(self.directory))
        else:
            config = Config2D(n_rays=8, **config)
            model = StarDist2D(config, name='model', basedir=str(self.directory))
        model.prepare_for_training()
        model.thresholds = dict(prob=0.99, nms=0.4)
        # Avoid the independent receptive-field calibration forwards in call-count tests.
        model._tile_overlap = [(0, 0)] * dimensions
        return model

    def callback(self, model, options=None, previews=0):
        shape = (6, 40, 40) if model.config.n_dim == 3 else (40, 40)
        label = np.zeros(shape, dtype=np.uint32)
        label[(slice(None),) * (len(shape) - 2) + (slice(5, 30), slice(5, 30))] = 70000
        image = np.random.RandomState(4).rand(*shape, 1).astype(np.float32)
        options = dict(minimum_batches=2, minimum_samples=4, **(options or {}))
        cb = validation.PatchValidation(model, [image], [label], model.logdir,
             options, previews, lambda **event: self.events.append(event), lambda: False, np.save)
        cb.set_model(model.keras_model)
        return cb

    def test_periodic_manual_phase_and_one_shot(self):
        request = self.directory / 'request'
        request.mkdir()
        scheduler = validation.FullValidationSchedule(5, request)
        self.assertIsNone(scheduler.consume(4))
        self.assertEqual('periodic', scheduler.consume(5))
        scheduler.started(5)
        (request / 'first.request').touch()
        self.assertEqual('requested', scheduler.consume(8))
        scheduler.started(8)
        self.assertFalse(list(request.iterdir()))
        self.assertIsNone(scheduler.consume(10))
        self.assertEqual('periodic', scheduler.consume(13))
        scheduler.started(13)
        self.assertEqual(18, scheduler.next_epoch)
        scheduler = validation.FullValidationSchedule(0, request)
        (request / 'second.request').touch()
        self.assertEqual('requested', scheduler.consume(8))
        scheduler.started(8)
        self.assertIsNone(scheduler.consume(9))
        self.assertIsNone(scheduler.consume(100))

    def test_object_metric_pools_objects_not_image_averages(self):
        target = np.array([[0, 1, 0, 2, 0, 3]], dtype=np.uint32)
        prediction = np.array([[0, 1, 0, 0, 0, 0]], dtype=np.uint32)
        counts = validation.object_counts(target, prediction)
        np.testing.assert_array_equal(counts, [1, 0, 2])
        self.assertEqual(0.5, validation.object_f1(counts))
        counts += validation.object_counts(target, target)
        self.assertAlmostEqual(0.8, validation.object_f1(counts))

    def test_request_during_active_pass_survives_and_is_deduplicated(self):
        directory = self.directory / 'control'
        directory.mkdir()
        scheduler = validation.FullValidationSchedule(5, directory, lambda **e: self.events.append(e))
        (directory / 'first.request').touch()
        scheduler.consume(8)
        scheduler.started(8)
        (directory / 'second.request').touch()
        scheduler.finish(8, 'completed')
        self.assertEqual(['second'], scheduler.pending)
        (directory / 'first.request').touch()
        scheduler.poll()
        self.assertEqual(['second'], scheduler.pending)
        self.assertEqual('requested', scheduler.consume(9))
        scheduler.started(9)
        self.assertEqual(14, scheduler.next_epoch)

    def test_planner_relaxes_sparse_foreground_and_covers_sources(self):
        labels = []
        for i in range(4):
            y = np.zeros((8, 64, 64), np.uint16)
            y[::4, ::16, ::16] = i + 1
            labels.append(y)
        images = [y[..., None].astype(np.float32) for y in labels]
        plan, summary, _ = build_plan(images, labels, (4, 16, 16), (1, 1, 1), 20, {},
                                     np.random.RandomState(42), lambda: None, self.directory / 'absent')
        self.assertEqual(20, len(set(plan)))
        self.assertFalse(summary['small_capacity'])
        self.assertLess(summary['minimum_foreground'], 0.01)
        self.assertEqual(7, summary['achieved_foreground_samples'])
        self.assertGreaterEqual(summary['achieved_foreground_sources'], 2)
        for i, (source, start) in enumerate(plan):
            for other_source, other in plan[:i]:
                if source == other_source:
                    self.assertTrue(any(abs(a - b) >= p for a, b, p in zip(start, other, (4, 16, 16))))

    def test_partial_final_batch_and_compatible_plan_reuse(self):
        model = self.model(batch=2)
        image = np.zeros((4, 16, 48, 1), np.float32)
        label = np.zeros(image.shape[:-1], np.uint16)
        args = (model, [image], [label], model.logdir, {}, 0, lambda **e: None, lambda: False, np.save)
        cb = validation.PatchValidation(*args)
        cb.set_model(model.keras_model)
        self.assertEqual(3, cb.sample_count)
        with patch.object(model.keras_model, 'call', wraps=model.keras_model.call) as forward:
            cb.on_epoch_end(0, {})
            self.assertEqual(2, forward.call_count)
        again = validation.PatchValidation(*args)
        self.assertEqual(cb.plan, again.plan)
        self.assertTrue(again.plan_summary['reused'])

    def test_object_extent_estimation_uses_bounded_deterministic_sample(self):
        labels = [np.zeros((8, 24, 24), np.int32) for _ in range(6)]
        for label in labels:
            label[2:6, 4:8, 3:9] = 1
            label[2:6, 14:18, 13:19] = 2
        np.testing.assert_array_equal([4, 4, 6], median_object_extents(labels, maximum=3))
        np.testing.assert_array_equal(median_object_extents(labels, maximum=3),
                                      median_object_extents(labels, maximum=3))
        self.assertIsNone(median_object_extents([np.zeros_like(labels[0])]))

    def test_disk_backed_sources_and_training_only_holdout_normalization(self):
        from tifffile import imwrite, TiffFile
        image = np.arange(8 * 32 * 32, dtype=np.uint16).reshape(8, 32, 32)
        label = np.zeros_like(image)
        label[:, 4:28, 4:28] = 700
        paths = [self.directory / name for name in ('image.tif', 'mask.tif')]
        for path, array in zip(paths, (image, label)):
            imwrite(path, array, photometric='minisblack', compression='deflate')
        calls = []
        def read(path, is_mask=False):
            calls.append(path)
            with TiffFile(path) as tiff:
                return tiff.series[0].asarray(out=store.decode_path(path)), 'ZYX'
        store = DatasetStore(read, lambda a, axes, mask: a if mask else a[..., None], 3, 1,
                             dict(cache_mb=0, statistics_cache_mb=0), lambda **e: None, lambda: False)
        self.addCleanup(store.close)
        x, y = store.pairs([paths])
        self.assertEqual([paths[1]], calls)  # No image decode during mask inspection.
        train_x, train_y, val_x, val_y = store.spatial_split(x, y, (4, 16, 16), .25)
        first = train_x[0][:4, :16, :16]
        _ = val_x[0][:4, :16, :16]
        self.assertEqual(0, store.bytes)
        self.assertEqual(0, store.analysis_bytes)
        self.assertEqual(2, len(calls))
        self.assertEqual(train_x[0].source.fit_normalization(), val_x[0].source.fit_normalization())
        self.assertEqual(train_y[0].source.analysis['objects'][700][0], np.count_nonzero(train_y[0][:]))
        self.assertEqual(train_y[0].source.analysis['objects'][700][0], np.count_nonzero(train_y[0][:]))
        self.assertEqual(2, len(calls))
        self.assertEqual(np.float32, first.dtype)
        self.assertEqual((4, 16, 16, 1), first.shape)
        self.assertTrue(all(isinstance(store.open_mapping(key).base, np.memmap) for key in store.mapped))

    def test_full_failure_preserves_checkpoints_and_does_not_abort_training(self):
        model = self.model()
        cb = self.callback(model, dict(full_every=1))
        with patch.object(cb, '_full_prediction', side_effect=MemoryError('test budget')):
            validation.train_with_validation(model, cb.images, cb.labels, cb, epochs=2, steps=1)
        self.assertEqual(2, len(cb.history))
        self.assertTrue((model.logdir / 'weights_best.h5').is_file())
        self.assertTrue((model.logdir / 'weights_last.h5').is_file())
        self.assertEqual(2, sum(e.get('info', {}).get('status') == 'failed' for e in self.events))

    def test_defaults_resolve_real_sample_budget(self):
        for batch, expected in [(1, 100), (2, 100), (4, 200)]:
            model = self.model(batch=batch)
            cb = self.callback(model, dict(minimum_foreground=0.001))
            cb = validation.PatchValidation(model, cb.images, cb.labels, model.logdir,
                    {}, 0, lambda **event: None, lambda: False, np.save)
            self.assertEqual(expected, cb.plan_summary['requested_samples'])
            self.assertEqual(4, cb.sample_count)
            self.assertEqual(4, len(set(cb.plan)))
            self.assertTrue(cb.plan_summary['small_capacity'])

    def test_reuses_predictions_and_loss_matches_keras(self):
        model = self.model()
        cb = self.callback(model)
        logs = {'loss': 99.0}
        before = np.random.get_state()
        python_state = random.getstate()
        with patch.object(model.keras_model, 'call', wraps=model.keras_model.call) as forward:
            cb.on_epoch_end(0, logs)
            self.assertEqual(4, forward.call_count)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(python_state, random.getstate())
        self.assertEqual(99.0, logs['loss'])
        self.assertTrue(np.isfinite(logs['val_loss']))
        inputs, targets, _ = cb._batch(0, 4)
        reference = model.keras_model.evaluate(inputs, targets, batch_size=1, verbose=0, return_dict=True)
        self.assertAlmostEqual(reference['loss'], logs['val_loss'], places=5)
        self.assertIn('val_object_f1', logs)

    def test_four_enlarged_previews_need_only_twelve_extra_tiles(self):
        model = self.model()
        cb = self.callback(model, previews=4)
        cb.plan = [(0, (0, y, x)) for y in (0, 16) for x in (0, 16)]
        with patch.object(model.keras_model, 'call', wraps=model.keras_model.call) as forward:
            cb.on_epoch_end(0, {})
            self.assertEqual(4, forward.call_count)
            cb.finish_epoch(1)
            self.assertEqual(16, forward.call_count)
        manifest = json.loads((model.logdir / 'previews/latest.json').read_text())
        self.assertEqual(4, len(manifest['samples']))
        for sample in manifest['samples']:
            label = np.load(sample['label_path'])
            self.assertEqual((4, 32, 32), label.shape)
            self.assertEqual(70000, label.max())
            self.assertTrue(Path(sample['prediction_path']).is_file())

    def test_empty_small_capacity_keeps_real_background(self):
        model = self.model()
        image = np.zeros((4, 16, 16, 1), dtype=np.float32)
        label = np.zeros(image.shape[:-1], dtype=np.uint16)
        cb = validation.PatchValidation(model, [image], [label], model.logdir, {}, 0,
                                        lambda **event: None, lambda: False, np.save)
        self.assertEqual(1, cb.sample_count)
        self.assertTrue(cb.plan_summary['small_capacity'])

    def test_enlarged_overlap_stitches_maps_in_their_original_coordinates(self):
        model = self.model()
        cb = self.callback(model)
        model.config.grid = (1, 2, 2)
        model._tile_overlap = [(0, 0), (2, 2), (2, 2)]
        cb.plan[0] = (0, (0, 4, 8))
        def forward(inputs, training=False):
            values = validation.tf.convert_to_tensor(inputs[:, :, ::2, ::2, :1])
            return values, validation.tf.repeat(values, 8, axis=-1)
        start = cb.plan[0][1]
        region = tuple(slice(s, s + p) for s, p in zip(start, cb.patch))
        anchor = [a.numpy()[0] for a in forward(cb.images[0][region][None])]
        with patch.object(model.keras_model, 'call', side_effect=forward):
            image, label, raw, geometry = cb._enlarge(0, anchor)
        self.assertEqual([1, 2, 2], geometry['tile_layout'])
        np.testing.assert_allclose(raw[0], image[:, ::2, ::2, :1], atol=1e-6)

    def test_two_dimensional_budget_and_checkpoint_training(self):
        model = self.model(dimensions=2)
        cb = self.callback(model, previews=1)
        self.assertEqual(4, cb.sample_count)
        validation.train_with_validation(model, cb.images, cb.labels, cb, epochs=2, steps=1)
        self.assertEqual(2, len(cb.history))
        self.assertTrue((model.logdir / 'weights_best.h5').is_file())
        self.assertTrue((model.logdir / 'weights_last.h5').is_file())
        self.assertFalse(any(event.get('info', {}).get('status') == 'started' and
                         event.get('info', {}).get('type') == 'full_validation' for event in self.events))

    def test_two_dimensional_explicit_budget_uses_shuffled_source_cycles(self):
        model = self.model(dimensions=2)
        images = np.random.RandomState(7).rand(4, 40, 40, 1).astype(np.float32)
        labels = np.zeros((4, 40, 40), np.int32)
        labels[:3, 5:30, 5:30] = 1  # Include a genuine background-only image.
        for budget in (2, 4, 11):
            with self.subTest(budget=budget):
                model.config.train_n_val_patches = budget
                args = (model, images, labels, model.logdir, {}, 0,
                        lambda **event: None, lambda: False, np.save)
                cb = validation.PatchValidation(*args)
                indices = [i for i, _ in cb.plan]
                self.assertEqual(budget, len(indices))
                for first in range(0, len(indices), len(images)):
                    cycle = indices[first:first + len(images)]
                    self.assertEqual(len(cycle), len(set(cycle)))
                    if len(cycle) == len(images):
                        self.assertEqual(list(range(len(images))), sorted(cycle))
                counts = np.bincount(indices, minlength=len(images))
                self.assertLessEqual(int(counts.max() - counts.min()), 1)
                self.assertEqual(cb.plan, validation.PatchValidation(*args).plan)

    def test_two_dimensional_automatic_budget_weights_image_capacity(self):
        model = self.model(dimensions=2)
        labels = [np.zeros((size, size), np.int32) for size in (16, 32, 64)]
        images = [label[..., None].astype(np.float32) for label in labels]
        for options, expected in [({}, [1, 4, 16]),
                                  (dict(minimum_batches=1, minimum_samples=10), [1, 2, 7]),
                                  (dict(minimum_batches=1, minimum_samples=1), [1, 1, 1])]:
            with self.subTest(options=options):
                args = (model, images, labels, model.logdir, options, 0,
                        lambda **event: self.events.append(event), lambda: False, np.save)
                cb = validation.PatchValidation(*args)
                counts = np.bincount([i for i, _ in cb.plan], minlength=len(labels)).tolist()
                self.assertEqual(expected, counts)
                self.assertEqual(sum(expected), cb.sample_count)
                self.assertEqual(cb.sample_count, len(set(cb.plan)))
                self.assertEqual(cb.plan, validation.PatchValidation(*args).plan)
                self.assertEqual(expected, self.events[-1]['info']['source_patch_counts'])
                for i, start in cb.plan:
                    self.assertTrue(all(0 <= s <= n - p for s, n, p in zip(start, labels[i].shape, cb.patch)))

    def test_two_dimensional_automatic_budget_is_bounded_and_shape_first(self):
        class ShapeOnly:
            shape = (1600000, 1600000)
            def __getitem__(self, key):
                raise AssertionError('Planning decoded an image/mask crop')
        model = self.model(dimensions=2, batch=4)
        args = (model, [ShapeOnly()], [ShapeOnly()], model.logdir, {}, 0,
                lambda **event: None, lambda: False, np.save)
        cb = validation.PatchValidation(*args)
        self.assertEqual(200, cb.sample_count)
        self.assertEqual(200, len(set(cb.plan)))

    def test_two_dimensional_expansion_does_not_repeat_forced_foreground_crop(self):
        label = np.zeros((32, 32), np.int32)
        label[0, 0] = 1  # Only one crop can include this foreground pixel.
        plan = build_2d_plan([label], (16, 16), 100, 1.0, np.random.RandomState(42), lambda: None)
        self.assertEqual([(0, (0, 0))], plan)

    def test_two_dimensional_validation_keeps_coordinates_across_epochs(self):
        model = self.model(dimensions=2)
        cb = self.callback(model)
        original = list(cb.plan)
        with patch.object(validation, 'sample_start', side_effect=AssertionError('Validation resampled a crop')):
            cb.on_epoch_end(0, {})
            cb.on_epoch_end(1, {})
        self.assertEqual(original, cb.plan)
        self.assertEqual(cb.history[0]['losses'], cb.history[1]['losses'])

    def test_three_dimensional_training_and_cancel(self):
        model = self.model()
        cb = self.callback(model)
        validation.train_with_validation(model, cb.images, cb.labels, cb, epochs=2, steps=1)
        self.assertEqual(2, len(cb.history))
        self.assertTrue((model.logdir / 'weights_best.h5').is_file())
        cb.cancelled = lambda: True
        with self.assertRaises(validation.ValidationCancelled):
            cb.on_epoch_end(2, {})

    def test_checkpoint_events_follow_native_saves_and_resume_best(self):
        model = self.model(dimensions=2)
        cb = self.callback(model)
        checkpoint = next(c for c in model.callbacks if getattr(c, 'save_best_only', False))
        checkpoint.filepath = str(model.logdir / 'custom_best.h5')
        checkpoint.set_model(model.keras_model)
        artifacts = validation.ValidationArtifacts(cb)
        artifacts.set_model(model.keras_model)
        def record(**event):
            if event.get('info', {}).get('type') == 'checkpoint':
                self.assertTrue(Path(event['info']['path']).is_file())
            self.events.append(event)
        cb.update = record
        with patch.object(cb, 'finish_epoch'), patch.object(cb, 'save_state'), \
                patch.object(validation.tf.train, 'Checkpoint'), \
                patch.object(model.keras_model, 'save_weights', wraps=model.keras_model.save_weights) as save:
            for epoch, loss in enumerate((3.0, 3.0, 4.0, 2.0)):
                if epoch == 2:
                    artifacts = validation.ValidationArtifacts(cb)
                    artifacts.set_model(model.keras_model)
                checkpoint.on_epoch_end(epoch, {'val_loss': loss})
                artifacts.on_epoch_end(epoch, {'val_loss': loss})
            # Two native best saves and four existing last saves; reporting adds none.
            self.assertEqual(6, save.call_count)
        checkpoints = [event['info'] for event in self.events if event.get('info', {}).get('type') == 'checkpoint']
        best = [event for event in checkpoints if event['kind'] == 'best']
        self.assertEqual([1, 4], [event['epoch'] for event in best])
        self.assertEqual([False, True], [event['overwrite'] for event in best])
        self.assertEqual([3.0, 2.0], [event['value'] for event in best])
        self.assertTrue(all(event['path'] == checkpoint.filepath for event in best))
        last = [event for event in checkpoints if event['kind'] == 'last']
        self.assertEqual([1, 2, 3, 4], [event['epoch'] for event in last])
        self.assertEqual([False, True, True, True], [event['overwrite'] for event in last])
        self.assertTrue(all(event['status'] == 'saved' for event in checkpoints))

    def test_checkpoint_precedes_full_validation_and_preview_retention_is_bounded(self):
        model = self.model()
        cb = self.callback(model, dict(full_every=1), previews=1)
        observed_epochs = []
        def full(epoch):
            self.assertTrue((model.logdir / 'weights_best.h5').is_file())
            self.assertTrue((model.logdir / 'weights_last.h5').is_file())
            observed_epochs.append(epoch)
        with patch.object(cb, '_full_validation', side_effect=full):
            validation.train_with_validation(model, cb.images, cb.labels, cb, epochs=3, steps=1)
        self.assertEqual([1, 2, 3], observed_epochs)
        self.assertFalse((model.logdir / 'previews/epoch_0001').exists())
        self.assertTrue((model.logdir / 'previews/epoch_0002').is_dir())
        self.assertTrue((model.logdir / 'previews/epoch_0003').is_dir())

    def test_full_volume_metric_is_separate_from_regular_validation(self):
        model = self.model()
        cb = self.callback(model)
        cb.on_epoch_end(0, {})
        regular = dict(cb.history[0])
        cb.full_reason = 'requested'
        cb._full_validation(1)
        self.assertEqual(regular['losses'], cb.history[0]['losses'])
        self.assertEqual(regular['object_f1'], cb.history[0]['object_f1'])
        self.assertNotIn('full_validation', cb.history[0])
        full = json.loads((model.logdir / 'full_validation_metrics.json').read_text())
        self.assertEqual(1, len(full['history'][0]['cases']))
        self.assertTrue(any(event.get('info', {}).get('status') == 'completed' for event in self.events))

    def test_crop_first_full_prediction_matches_native_sparse_tiling(self):
        model = self.model()
        cb = self.callback(model)
        model.thresholds = dict(prob=0.5, nms=0.4)
        image = cb.images[0]
        native = model.predict_sparse(image, axes='ZYXC', normalizer=None,
                                      n_tiles=(2, 3, 3, 1), show_tile_progress=False)
        captured = {}
        def reconstruct(shape, probabilities, distances, **kwargs):
            captured.update(prob=probabilities, dist=distances, points=kwargs['points'])
            return np.zeros(shape, np.int32), {}
        with patch.object(model, '_instances_from_prediction', side_effect=reconstruct):
            cb._full_prediction(image, cb.labels[0].shape, 1, 0)
        np.testing.assert_allclose(captured['prob'], native[0], atol=1e-6)
        np.testing.assert_allclose(captured['dist'], native[1], atol=1e-6)
        np.testing.assert_array_equal(captured['points'], native[2])

    def test_resume_reuses_coordinates_optimizer_and_full_validation_phase(self):
        model = self.model()
        cb = self.callback(model, dict(full_every=1))
        validation.train_with_validation(model, cb.images, cb.labels, cb, epochs=1, steps=1)
        resumed = self.callback(model, dict(full_every=1, resume=True))
        self.assertEqual(cb.plan, resumed.plan)
        self.assertTrue(resumed.plan_summary['reused'])
        self.assertEqual(1, resumed.initial_epoch)
        self.assertEqual(2, resumed.schedule.next_epoch)
        validation.train_with_validation(model, resumed.images, resumed.labels, resumed, epochs=2, steps=1)
        self.assertEqual(2, int(model.keras_model.optimizer.iterations.numpy()))
        self.assertEqual([1, 2], [entry['epoch'] for entry in resumed.history])

    def test_invalid_resume_preserves_saved_plan(self):
        model = self.model()
        cb = self.callback(model)
        plan_path = model.logdir / 'validation_plan.json'
        original = plan_path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'without saved validation state'):
            self.callback(model, dict(resume=True))
        self.assertEqual(original, plan_path.read_bytes())
        cb.save_state(1)
        with self.assertRaisesRegex(ValueError, 'incompatible validation plan'):
            self.callback(model, dict(resume=True, seed=123))
        self.assertEqual(original, plan_path.read_bytes())

    def test_request_after_final_boundary_is_reported_and_cleared(self):
        model = self.model()
        cb = self.callback(model)
        request = self.directory / 'full-validation'
        request.mkdir()
        cb.schedule.request_path = request

        class LateRequest(validation.Callback):
            def on_epoch_end(self, epoch, logs=None):
                (request / 'late.request').touch()

        model.callbacks.append(LateRequest())
        with patch.object(cb, '_full_validation') as full:
            validation.train_with_validation(model, cb.images, cb.labels, cb, epochs=1, steps=1)
            full.assert_not_called()
        self.assertFalse(list(request.iterdir()))
        self.assertTrue(any('after the final epoch boundary' in event.get('message', '')
                            for event in self.events))
        self.assertEqual('closed', self.events[-1]['info']['status'])

    def test_generated_java_training_tasks(self):
        tasks = json.loads(os.environ.get('JDLL_GENERATED_TRAINING_TASKS', '[]'))
        if not tasks:
            self.skipTest('Generated task fixtures are supplied by StarDistValidationTest in Java')
        from tifffile import imwrite
        from types import SimpleNamespace
        for fixture in tasks:
            self.events.clear()
            dimensions = fixture['dimensions']
            shape = (6, 40, 40) if dimensions == 3 else (40, 40)
            image = np.random.RandomState(5).rand(*shape).astype(np.float32)
            label = np.zeros(shape, dtype=np.uint16)
            label[(slice(None),) * (dimensions - 2) + (slice(8, 28), slice(8, 28))] = 1
            for split in (('train',) if fixture.get('single_source') else ('train', 'val')):
                images = Path(fixture['dataset']) / split / 'images'
                masks = Path(fixture['dataset']) / split / 'masks'
                images.mkdir(parents=True)
                masks.mkdir(parents=True)
                for directory, array in ((images, image), (masks, label)):
                    imwrite(directory / 'sample.tif', array, photometric='minisblack', compression='deflate',
                            metadata={'axes': 'ZYX' if dimensions == 3 else 'YX'})
            task = SimpleNamespace(outputs={}, update=lambda **event: self.events.append(event))
            # Exercise the whole generated task, not just the standalone validation helper.
            scope = {'task': task}
            code = Path(fixture['script']).read_text()
            exec(compile(code, fixture['script'], 'exec'), scope)
            output = Path(fixture['output'])
            self.assertEqual(str(output), task.outputs['result'])
            self.assertTrue((output / 'weights_best.h5').is_file())
            self.assertTrue((output / 'weights_last.h5').is_file())
            self.assertTrue((output / 'previews/latest.json').is_file())
            self.assertTrue(any('val/object_f1' in event.get('info', {}).get('metrics', {}) for event in self.events))


if __name__ == '__main__':
    unittest.main()
