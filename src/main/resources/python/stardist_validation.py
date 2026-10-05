"""JDLL-owned StarDist validation. Uses StarDist's targets, losses and NMS."""

import itertools
import json
import math
import os
import random
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from csbdeep.utils.tf import IS_KERAS_3_PLUS, keras_import
from csbdeep.internals.predict import Tiling
from stardist.matching import matching
from stardist.nms import _ind_prob_thresh
from stardist.models.model2d import StarDistData2D
from stardist.models.model3d import StarDistData3D
from stardist.rays3d import rays_from_json
from stardist_data import sample_start, available_memory
from stardist_sampling import build_2d_plan, build_plan
from validation_control import FullValidationSchedule

Callback = keras_import('callbacks', 'Callback')


class ValidationCancelled(Exception):
    pass


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, allow_nan=False)
    os.replace(temporary, path)


def data_generator(model, images, labels, batch_size, length, **overrides):
    config = model.config
    kwargs = dict(grid=config.grid, patch_size=config.train_patch_size,
                  use_gpu=False, foreground_prob=config.train_foreground_only,
                  n_classes=config.n_classes, sample_ind_cache=config.train_sample_cache)
    if config.n_dim == 3:
        generator = StarDistData3D
        kwargs.update(rays=rays_from_json(config.rays_json), anisotropy=config.anisotropy)
    else:
        generator = StarDistData2D
        kwargs.update(n_rays=config.n_rays, shape_completion=config.train_shape_completion,
                      b=config.train_completion_crop)
    kwargs.update(overrides)
    if IS_KERAS_3_PLUS:
        kwargs['keras_kwargs'] = dict(workers=0, use_multiprocessing=False)
    classes = model._parse_classes_arg('auto', len(images))
    return generator(images, labels, batch_size=batch_size, length=length, classes=classes, **kwargs)


def crop_batch(model, images, labels, plan):
    """Read only selected crops, then delegate target construction to StarDist."""
    xs, ys = [], []
    for index, start in plan:
        region = tuple(slice(s, s + p) for s, p in zip(start, model.config.train_patch_size))
        ys.append(labels[index][region])
        xs.append(images[index][region])
    generator = data_generator(model, xs, ys, len(xs), 1, foreground_prob=0, sample_ind_cache=False)
    generator.batch = lambda i: np.arange(len(xs))
    inputs, targets = generator[0]
    border = model.config.train_completion_crop if model.config.n_dim == 2 and model.config.train_shape_completion else 0
    if border:
        ys = [y[border:-border, border:-border] for y in ys]
    return inputs, targets, ys


def object_counts(target, prediction):
    if np.any(target < 0):
        prediction = np.where(target >= 0, prediction, 0)
        target = np.maximum(target, 0)
    result = matching(target, prediction, thresh=0.5, criterion='iou')
    return np.array([result.tp, result.fp, result.fn], dtype=np.int64)


def object_f1(counts):
    tp, fp, fn = (int(value) for value in counts)
    denominator = 2 * tp + fp + fn
    return 2.0 * tp / denominator if denominator else 0.0


class PatchValidation(Callback):
    def __init__(self, model_ref, images, labels, output_dir, options, preview_count,
                 update, cancelled, save_array, source_paths=None, request_path=None):
        super().__init__()
        self.ref = model_ref
        self.images, self.labels = images, labels
        self.output = Path(output_dir)
        self.options = dict(options)
        self.update, self.cancelled, self.save_array = update, cancelled, save_array
        self.sources = source_paths or [str(i) for i in range(len(images))]
        self.dim = model_ref.config.n_dim
        self.patch = tuple(model_ref.config.train_patch_size)
        self.batch_size = int(model_ref.config.train_batch_size)
        self.preview_count = max(0, int(preview_count))
        large = (getattr(model_ref.config, 'unet_n_filter_base', 32) > 32
                 or (self.dim == 3 and getattr(model_ref.config, 'unet_n_depth', 2) > 2))
        default_preview_mb = 600 if large else 256
        preview_bytes = self.options.get('preview_max_bytes', 'auto')
        self.preview_bytes = default_preview_mb * 1024**2 if preview_bytes in ('auto', None) else int(preview_bytes)
        if self.preview_bytes <= 0:
            raise ValueError('validation.preview_max_bytes must be positive.')
        self.schedule = FullValidationSchedule(self.options.get('full_every', 0), request_path, update)
        self.history = []
        self.initial_epoch = 0
        self.preview_samples = []
        self.preview_directories = []
        self.full_reason = None
        self.rng = np.random.RandomState(int(self.options.get('seed', 42)))
        self.sampling_lock = threading.Lock()
        self.anchors = []
        self.plan_summary = {}
        self.signature = None
        if self.dim == 3:
            self.update(message='Preparing fixed 3D validation patches.', info={'type': 'validation_plan'})
            batches = int(self.options.get('minimum_batches', 50))
            samples = int(self.options.get('minimum_samples', 100))
            if min(batches, samples, self.batch_size) < 1:
                raise ValueError('Validation batch and sample budgets must be positive.')
            requested = max(batches, int(math.ceil(samples / self.batch_size))) * self.batch_size
            plan_options = dict(self.options, resolved_context=dict(axes=model_ref.config.axes,
                channels=model_ref.config.n_channel_in, anisotropy=model_ref.config.anisotropy,
                normalization='per_channel_1_99.8', rays=model_ref.config.rays_json))
            self.plan, self.plan_summary, self.signature = build_plan(images, labels, self.patch,
                model_ref.config.grid, requested, plan_options, self.rng, self.check_cancel,
                self.output / 'validation_plan.json')
            self.sample_count = len(self.plan)
            self.plan_summary.update(requested_batches=int(math.ceil(requested / self.batch_size)),
                                     achieved_batches=int(math.ceil(self.sample_count / self.batch_size)))
            if self.options.get('resume'):
                state_path = self.output / 'validation_state.json'
                if not state_path.is_file():
                    raise ValueError('Cannot resume without saved validation state.')
                saved = json.loads(state_path.read_text())
                if saved.get('signature') != self.signature:
                    raise ValueError('Cannot resume with an incompatible validation plan.')
                if saved['schedule']['interval'] != self.schedule.interval:
                    raise ValueError('Cannot resume with a changed full-validation interval.')
                self.schedule.next_epoch = saved['schedule']['next_epoch']
                self.initial_epoch = int(saved['epoch'])
                self.resume_checkpoint = saved['training_checkpoint']
                metrics = self.output / 'validation_metrics.json'
                self.history = [v for v in json.loads(metrics.read_text())['history']
                                if v['epoch'] <= self.initial_epoch] if metrics.exists() else []
            atomic_json(self.output / 'validation_plan.json', dict(signature=self.signature,
                sources=self.sources, source_regions=[getattr(getattr(y, 'source', None), 'bounds', None) for y in labels],
                patch_size=self.patch, patches=self.plan, summary=self.plan_summary))
        elif model_ref.config.train_n_val_patches is None:
            batches = int(self.options.get('minimum_batches', 50))
            samples = int(self.options.get('minimum_samples', 100))
            if min(batches, samples, self.batch_size) < 1:
                raise ValueError('Validation batch and sample budgets must be positive.')
            requested = max(len(labels), max(batches, int(math.ceil(samples / self.batch_size))) * self.batch_size)
            self.plan = build_2d_plan(labels, self.patch, requested,
                model_ref.config.train_foreground_only, self.rng, self.check_cancel)
            self.sample_count = len(self.plan)
            self.update(message='Automatic 2D validation: %d patches from %d images (target %d); '
                        'extra patches weighted by image capacity, limited by available area and unique sampled coordinates.' %
                        (self.sample_count, len(labels), requested),
                        info=dict(type='validation_plan', requested_samples=requested, achieved_samples=self.sample_count,
                                  sampling='image_capacity', source_patch_counts=np.bincount(
                                      [i for i, _ in self.plan], minlength=len(labels)).tolist()))
        else:
            # Visit every image before repeating; fix the sampled crops for all epochs.
            self.sample_count = int(model_ref.config.train_n_val_patches)
            if self.sample_count < 1:
                raise ValueError('train_n_val_patches must be positive or None for automatic validation.')
            self.plan = []
            for first in range(0, self.sample_count, len(labels)):
                for index in self.rng.permutation(len(labels))[:self.sample_count - first]:
                    i = int(index)
                    self.plan.append((i, sample_start(labels[i], self.patch, self.rng,
                                     self.rng.rand() < model_ref.config.train_foreground_only)))
        self.anchor_indices = set(np.linspace(0, self.sample_count - 1,
                                  min(self.preview_count, self.sample_count), dtype=int).tolist())
        atomic_json(self.output / 'validation_config.json', dict(
            self.options, dimensions=self.dim, batch_size=self.batch_size,
            samples=self.sample_count, batches=int(math.ceil(self.sample_count / self.batch_size)),
            checkpoint_monitor='val_loss', object_metric='Object F1 at IoU >= 0.5',
            aggregation='pooled_tp_fp_fn', preview_count=self.preview_count,
            preview_max_bytes=self.preview_bytes, full_every=self.schedule.interval))

    def check_cancel(self):
        if self.cancelled():
            raise ValidationCancelled()

    @contextmanager
    def _sampling_rng(self):
        # Native target generators use global RNGs even for exact-sized crops.
        # Exclude Keras's training prefetch while temporarily preserving those states.
        with self.sampling_lock:
            numpy_state, python_state = np.random.get_state(), random.getstate()
            try:
                yield
            finally:
                np.random.set_state(numpy_state)
                random.setstate(python_state)

    def _batch(self, first, last):
        with self._sampling_rng():
            return crop_batch(self.ref, self.images, self.labels, self.plan[first:last])

    def _instances(self, shape, arrays):
        kwargs = {'prob_class': arrays[2]} if len(arrays) > 2 else {}
        return self.ref._instances_from_prediction(shape, arrays[0][..., 0],
                    np.maximum(arrays[1], 1e-3), **kwargs)[0]

    def on_train_begin(self, logs=None):
        self.update(message='StarDist validation: %d patches, %d batches; Object F1 at IoU >= 0.5 '
                    '(pooled objects). Best checkpoint uses validation loss.' %
                    (self.sample_count, math.ceil(self.sample_count / self.batch_size)),
                    info=dict(self.plan_summary, type='validation_plan', samples=self.sample_count))
        if self.plan_summary:
            s = self.plan_summary
            self.update(message='Validation plan: requested %d patches/%d batches; achieved %d patches/%d batches '
                        'from %d/%d sources; forced foreground %d/%d patches from %d/%d requested sources; '
                        'minimum foreground %.4f%%, sampling overlap at most %.1f%%.' %
                        (s['requested_samples'], s['requested_batches'], s['achieved_samples'], s['achieved_batches'],
                         s['sources_covered'], s['eligible_sources'], s['achieved_foreground_samples'],
                         s['requested_foreground_samples'], s['achieved_foreground_sources'],
                         s['requested_foreground_sources'], 100 * s['minimum_foreground'], 100 * s['maximum_overlap']),
                        info=dict(s, type='validation_plan'))
        for reason in self.plan_summary.get('fallback_reasons', []):
            self.update(message='Validation plan: ' + reason + '.', info={'type': 'validation_plan'})
        self.schedule.emit('ready', supported=self.dim == 3)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.preview_samples = []
        self.anchors = []
        self.check_cancel()
        self.full_reason = self.schedule.consume(epoch + 1) if self.dim == 3 else None
        self.update(message='Starting patch validation for epoch %d (%d patches).' % (epoch + 1, self.sample_count),
                    info={'type': 'validation', 'status': 'started', 'epoch': epoch + 1})
        counts = np.zeros(3, dtype=np.int64)
        loss_sums = {}
        retained_bytes = 0
        try:
            for first in range(0, self.sample_count, self.batch_size):
                self.check_cancel()
                self.schedule.poll()
                last = min(first + self.batch_size, self.sample_count)
                inputs, targets, masks = self._batch(first, last)
                predictions = self.model(inputs, training=False)
                targets = tuple(tf.convert_to_tensor(y) for y in targets)
                # Use the functions installed by StarDist.prepare_for_training, without a
                # second forward pass or resetting Keras's training metric accumulators.
                components = [float(tf.reduce_mean(loss(target, prediction)).numpy())
                              for loss, target, prediction in zip(self.model.loss, targets, predictions)]
                values = {name + '_loss': value for name, value in zip(self.model.output_names, components)}
                values['loss'] = sum(weight * value for weight, value in
                                     zip(self.ref.config.train_loss_weights, components))
                values['loss'] += sum(float(tf.reduce_sum(value).numpy()) for value in self.model.losses)
                for name, value in values.items():
                    loss_sums[name] = loss_sums.get(name, 0.0) + (last - first) * value
                arrays = [p.numpy() for p in predictions]
                for offset, mask in enumerate(masks):
                    raw = [p[offset] for p in arrays]
                    prediction = self._instances(mask.shape, raw)
                    counts += object_counts(mask, prediction)
                    sample_index = first + offset
                    if sample_index in self.anchor_indices:
                        anchor_bytes = inputs[0][offset].nbytes + mask.nbytes + prediction.nbytes + sum(a.nbytes for a in raw)
                        if retained_bytes + anchor_bytes <= self.preview_bytes:
                            # Copy only chosen anchors; do not pin their entire prediction batches.
                            self.anchors.append((sample_index, inputs[0][offset].copy(), mask.copy(),
                                                 [a.copy() for a in raw], prediction.copy()))
                            retained_bytes += anchor_bytes
                        else:
                            self.update(message='Skipping a validation preview to respect the preview memory budget.',
                                        info={'type': 'warning'})
                if last == self.sample_count or (first // self.batch_size + 1) % 5 == 0:
                    self.update(message='Validated %d/%d patches.' % (last, self.sample_count),
                                info={'type': 'validation', 'epoch': epoch + 1, 'samples': last,
                                      'total_samples': self.sample_count, 'completed': last,
                                      'total': self.sample_count, 'unit': 'patches', 'status': 'progress'})
            values = {name: value / self.sample_count for name, value in loss_sums.items()}
            if not values or not all(np.isfinite(value) for value in values.values()):
                raise ValueError('Non-finite or missing StarDist validation loss.')
            logs.update({'val_' + name: value for name, value in values.items()})
            logs['val_object_f1'] = object_f1(counts)
            self.history.append({'epoch': epoch + 1, 'losses': values, 'object_f1': logs['val_object_f1'],
                                 'tp': int(counts[0]), 'fp': int(counts[1]), 'fn': int(counts[2])})
            atomic_json(self.output / 'validation_metrics.json', {'history': self.history})
            self.update(info={'type': 'validation', 'status': 'completed', 'epoch': epoch + 1,
                              'metrics': {'object_f1': logs['val_object_f1']}, 'losses': values})
        except ValidationCancelled:
            raise
        except Exception as error:
            raise RuntimeError('StarDist validation failed: ' + str(error)) from error

    def _preview_estimate(self, sample_index, image, raw):
        shape = self._preview_geometry(sample_index)[2] if self.dim == 3 else image.shape[:-1]
        voxels = math.prod(shape)
        grid_voxels = math.prod(s // g for s, g in zip(shape, self.ref.config.grid))
        artifact_bytes = voxels * (image.shape[-1] * image.dtype.itemsize + 8) + grid_voxels * 4
        map_bytes = grid_voxels * sum(a.shape[-1] * a.dtype.itemsize for a in raw)
        # Only one region is stitched at a time. Do not charge all four ray-map
        # workspaces cumulatively; only their small saved image/label assets persist.
        workspace = artifact_bytes * 2 + map_bytes * 2 + 2 * sum(a.nbytes for a in raw) + grid_voxels * 8
        return workspace, artifact_bytes

    def _save_preview(self, epoch, sample_index, image, label, raw, prediction, enlarge=True):
        geometry = {}
        if self.dim == 3 and enlarge:
            image, label, raw, geometry = self._enlarge(sample_index, raw)
            prediction = self._instances(label.shape, raw)
        elif self.dim == 3:
            geometry = dict(region_start=self.plan[sample_index][1], shape=label.shape,
                            tile_layout=[1, 1, 1], additional_tiles=0)
        directory = self.output / 'previews' / ('epoch_%04d' % epoch)
        directory.mkdir(parents=True, exist_ok=True)
        sample = {'index': sample_index, 'axes': self.ref.config.axes, 'full_volume': False}
        sample.update(geometry)
        for key, array in [('image', image), ('label', label), ('prediction', prediction), ('prob', raw[0][..., 0])]:
            path = directory / ('preview_%03d_%s.npy' % (sample_index, key))
            self.save_array(path, array)
            sample[key + '_path'] = str(path)
        if self.dim == 3:
            index, start = self.plan[sample_index]
            sample['source_image'] = self.sources[index]
            sample['anchor_start'] = start
            if hasattr(self.images[index], 'source'):
                sample['source_region'] = getattr(self.images[index].source, 'bounds', None)
            occupied = np.count_nonzero(label > 0, axis=(1, 2))
            sample['initial_plane'] = {'axis': 'z', 'index': int(np.argmax(occupied)) if np.any(occupied) else label.shape[0] // 2}
        self._save_pngs(directory, sample, image, label, prediction)
        return sample

    def _save_pngs(self, directory, sample, image, label, prediction):
        from PIL import Image
        plane = sample.get('initial_plane', {}).get('index', 0)
        x, y, p = (image[plane], label[plane], prediction[plane]) if self.dim == 3 else (image, label, prediction)
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        elif x.shape[-1] == 2:
            x = np.concatenate((x, np.zeros_like(x[..., :1])), axis=-1)
        rgb = np.asarray(np.clip(x[..., :3], 0, 1) * 255, dtype=np.uint8)
        for key, values in [('image', rgb), ('target', y), ('prediction', p)]:
            if key != 'image':
                ids = np.maximum(values, 0).astype(np.uint64)
                color = np.stack([(ids * multiplier + 31) % 224 + 31 for multiplier in (53, 97, 193)], axis=-1)
                color[ids == 0] = 0
                values = np.where((ids > 0)[..., None], 0.5 * rgb + 0.5 * color, rgb).astype(np.uint8)
            path = directory / ('preview_%03d_%s.png' % (sample['index'], key))
            Image.fromarray(values).save(path)
            sample[key + '_png_path'] = str(path)

    def _preview_geometry(self, sample_index):
        index, start = self.plan[sample_index]
        overlap = self._tile_overlap()
        div = self.ref._axes_div_by('ZYX')
        positions = [[start[0]]]
        for axis in (1, 2):
            # StarDist's halo is needed on each side of an internal tile boundary.
            step = (self.patch[axis] - 2 * overlap[axis]) // div[axis] * div[axis]
            if step < div[axis]:
                positions.append([start[axis]])
                continue
            next_start = start[axis] + step
            if next_start + self.patch[axis] > self.labels[index].shape[axis]:
                next_start = start[axis] - step
            positions.append(sorted({start[axis], next_start}) if next_start >= 0 else [start[axis]])
        origin = tuple(p[0] for p in positions)
        shape = tuple(p[-1] + size - p[0] for p, size in zip(positions, self.patch))
        return positions, origin, shape

    def _tile_overlap(self):
        if not hasattr(self.ref, '_tile_overlap'):
            with self._sampling_rng():
                return self.ref._axes_tile_overlap('ZYX')
        return self.ref._axes_tile_overlap('ZYX')

    def _enlarge(self, sample_index, anchor):
        index, start = self.plan[sample_index]
        grid = tuple(self.ref.config.grid)
        positions, origin, shape = self._preview_geometry(sample_index)
        stitched = [np.zeros(tuple(s // g for s, g in zip(shape, grid)) + (a.shape[-1],), dtype=a.dtype) for a in anchor]
        weights = np.zeros(stitched[0].shape[:-1], np.float32)
        window = self._blend_window(anchor[0].shape[:-1])
        for position in itertools.product(*positions):
            self.check_cancel()
            region = tuple(slice(s, s + p) for s, p in zip(position, self.patch))
            if position == tuple(start):
                raw = anchor
            else:
                raw = [a.numpy()[0] for a in self.model(self.images[index][region][None], training=False)]
            destination = tuple(slice((at - base) // g, (at - base + p) // g)
                                for at, base, p, g in zip(position, origin, self.patch, grid))
            for target, values in zip(stitched, raw):
                target[destination] += values * window[..., None]
            weights[destination] += window
        for target in stitched:
            target /= weights[..., None]
        region = tuple(slice(s, s + size) for s, size in zip(origin, shape))
        layout = [len(p) for p in positions]
        return self.images[index][region], self.labels[index][region], stitched, {
            'region_start': origin, 'shape': shape, 'tile_layout': layout,
            'additional_tiles': int(np.prod(layout)) - 1}

    @staticmethod
    def _blend_window(shape):
        weights = np.ones(shape, np.float32)
        for axis, size in enumerate(shape):
            ramp = np.maximum(0.05, 1 - np.abs(np.linspace(-1, 1, size))).astype(np.float32)
            weights *= ramp.reshape((1,) * axis + (size,) + (1,) * (len(shape) - axis - 1))
        return weights

    def finish_epoch(self, epoch):
        self.check_cancel()
        try:
            while self.anchors:
                self.check_cancel()
                sample = self.anchors.pop(0)
                retained = sum(x.nbytes + y.nbytes + p.nbytes + sum(a.nbytes for a in raw)
                               for _, x, y, raw, p in self.anchors)
                workspace, _ = self._preview_estimate(sample[0], sample[1], sample[3])
                available = available_memory()
                budget = min(self.preview_bytes, available // 2) if available is not None else self.preview_bytes
                enlarge = workspace + retained <= budget
                if not enlarge:
                    anchor_workspace = (sample[1].nbytes + sample[2].nbytes + sample[4].nbytes
                                        + sum(a.nbytes for a in sample[3])) * 3
                    if anchor_workspace + retained > budget:
                        self.update(message='Skipped validation preview: insufficient working memory.',
                                    info={'type': 'preview', 'status': 'reduced', 'epoch': epoch})
                        continue
                    self.update(message='Reduced validation preview extent to respect its memory budget.',
                                info={'type': 'preview', 'status': 'reduced', 'epoch': epoch})
                self.preview_samples.append(self._save_preview(epoch, *sample, enlarge=enlarge))
        except ValidationCancelled:
            raise
        except Exception as error:
            self.update(message='Validation preview failed; regular checkpoints are preserved: ' + str(error),
                        info={'type': 'preview', 'status': 'failed', 'epoch': epoch})
        finally:
            self.anchors.clear()
        if self.preview_samples:
            path = self.output / 'previews' / 'latest.json'
            atomic_json(path, {'epoch': epoch, 'n_dim': self.dim, 'axes': self.ref.config.axes,
                               'anisotropy': getattr(self.ref.config, 'anisotropy', None),
                               'scope': 'validation_patches', 'samples': self.preview_samples})
            self.update(message='Saved validation patches for epoch %d at: %s' % (epoch, path),
                        info={'type': 'preview', 'epoch': epoch, 'preview_path': str(path)})
            if self.dim == 3:
                self.update(message='Validation previews: %d regions, %d extra tiles (excluded from loss and F1).'
                            % (len(self.preview_samples), sum(s['additional_tiles'] for s in self.preview_samples)),
                            info={'type': 'validation_previews'})
            self.preview_directories.append(Path(self.preview_samples[0]['image_path']).parent)
            # Keep the previous published assets while Swing loads the new manifest.
            # Never accumulate one set of volume arrays per training epoch.
            for directory in self.preview_directories[:-2]:
                try:
                    shutil.rmtree(directory)
                except OSError as error:
                    self.update(message='Could not remove older validation previews: ' + str(error),
                                info={'type': 'warning'})
            self.preview_directories = [p for p in self.preview_directories if p.exists()]
        if self.full_reason:
            self._full_validation(epoch)
        self.save_state(epoch)

    def save_state(self, epoch):
        atomic_json(self.output / 'validation_state.json', dict(epoch=epoch, signature=self.signature,
            schedule=self.schedule.state(), training_checkpoint='training_state-%06d' % epoch))
        for path in self.output.glob('training_state-*.*'):
            number = path.name.split('-', 1)[1].split('.', 1)[0]
            if number.isdigit() and int(number) < epoch - 1:
                path.unlink()

    def _full_validation(self, epoch):
        self.schedule.started(epoch)
        self.save_state(epoch)
        counts = np.zeros(3, dtype=np.int64)
        cases = []
        try:
            for i, (image, label) in enumerate(zip(self.images, self.labels)):
                self.check_cancel()
                prediction = self._full_prediction(image, label.shape, epoch, i)
                case_counts = object_counts(np.asarray(label), prediction)
                counts += case_counts
                cases.append(dict(source=self.sources[i], object_f1=object_f1(case_counts),
                                  source_region=getattr(getattr(label, 'source', None), 'bounds', None),
                                  tp=int(case_counts[0]), fp=int(case_counts[1]), fn=int(case_counts[2])))
                self.update(message='Full validation: %d/%d volumes.' % (i + 1, len(self.images)),
                            info={'type': 'full_validation', 'status': 'progress', 'epoch': epoch,
                                  'completed': i + 1, 'total': len(self.images), 'unit': 'volumes'})
            metrics = dict(epoch=epoch, object_f1=object_f1(counts), tp=int(counts[0]), fp=int(counts[1]), fn=int(counts[2]))
            path = self.output / 'full_validation_metrics.json'
            history = json.loads(path.read_text()).get('history', []) if path.exists() else []
            history.append(dict(metrics, status='completed', request_ids=list(self.schedule.active), cases=cases))
            atomic_json(path, dict(history=history))
            self.schedule.finish(epoch, 'completed', metrics=metrics, path=str(path),
                message='Full validation epoch %d: Object F1 at IoU >= 0.5 = %.5f.' % (epoch, metrics['object_f1']))
        except ValidationCancelled:
            self._record_full_failure(epoch, cases, 'cancelled', None)
            self.schedule.finish(epoch, 'cancelled')
            raise
        except Exception as error:
            self._record_full_failure(epoch, cases, 'failed', str(error))
            self.schedule.finish(epoch, 'failed', error=str(error),
                message='Full validation failed; regular checkpoints are preserved: ' + str(error))

    def _record_full_failure(self, epoch, cases, status, error):
        path = self.output / 'full_validation_metrics.json'
        try:
            payload = json.loads(path.read_text()) if path.exists() else {'history': []}
            payload['history'].append(dict(epoch=epoch, status=status, error=error, cases=cases))
            atomic_json(path, payload)
        except OSError as write_error:
            self.update(message='Could not save full-validation failure details: ' + str(write_error),
                        info={'type': 'warning'})

    def _full_prediction(self, image, shape, epoch, case):
        """Native halo/core tiling and sparse global NMS, without dense volume ray maps."""
        grid = self.ref.config.grid
        div = self.ref._axes_div_by('ZYX')
        padded = tuple(int(math.ceil(n / d)) * d for n, d in zip(shape, div))
        estimate = math.prod(shape) * 24
        maximum = int(self.options.get('full_max_bytes', 1024**3))
        available = available_memory()
        if available is not None:
            maximum = min(maximum, available // 2)
        if estimate > maximum:
            raise MemoryError('Full-volume reconstruction needs approximately %d MiB; validation.full_max_bytes '
                              'allows %d MiB. Regular patch validation/checkpoints are unaffected.' %
                              (estimate // 1024**2, maximum // 1024**2))
        axes = []
        for n, p, d, halo in zip(padded, self.patch, div, self._tile_overlap()):
            tiling = Tiling.for_n_tiles(n // d, int(math.ceil(n / p)), int(math.ceil(halo / d)))
            axes.append(list(tiling.slice_generator(d)))
        probabilities, distances, points, classes = [], [], [], []
        candidate_bytes = 0
        total = math.prod(len(a) for a in axes)
        for done, tile in enumerate(itertools.product(*axes), 1):
            self.check_cancel()
            self.schedule.poll()
            reads, writes, crops = zip(*tile)
            tile_shape = tuple(s.stop - s.start for s in reads)
            workspace = math.prod(tile_shape) * (image.shape[-1] * 4 + (self.ref.config.n_rays + 1) * 8)
            if estimate + candidate_bytes * 3 + workspace > maximum:
                raise MemoryError('Full-validation candidate/tile workspace exceeds validation.full_max_bytes.')
            region = tuple(slice(s.start, min(n, s.stop)) for s, n in zip(reads, shape))
            crop = image[region]
            padding = [(0, p - n) for n, p in zip(crop.shape[:-1], tile_shape)] + [(0, 0)]
            if any(hi for _, hi in padding):
                crop = np.pad(crop, padding, mode='reflect')
            raw = [a.numpy()[0] for a in self.model(crop[None], training=False)]
            selected = tuple(slice((s.start or 0) // g, (n if s.stop is None else
                             n + s.stop if s.stop < 0 else s.stop) // g)
                             for s, n, g in zip(crops, tile_shape, grid))
            prob = raw[0][selected][..., 0]
            border = [(2 if s.start == 0 else -1, 2 if s.stop == n else -1) for s, n in zip(writes, padded)]
            keep = _ind_prob_thresh(prob, self.ref.thresholds.prob, b=border)
            coords = np.column_stack(np.nonzero(keep))
            coords = (coords + np.array([s.start // g for s, g in zip(writes, grid)])) * np.array(grid)
            valid = np.all(coords < np.array(shape), axis=1)
            probabilities.append(prob[keep][valid])
            distances.append(np.maximum(raw[1][selected][keep][valid], 1e-3))
            points.append(coords[valid])
            candidate_bytes += probabilities[-1].nbytes + distances[-1].nbytes + points[-1].nbytes
            if len(raw) > 2:
                classes.append(raw[2][selected][keep][valid])
                candidate_bytes += classes[-1].nbytes
            if estimate + candidate_bytes * 3 + workspace > maximum:
                raise MemoryError('Full-validation candidates exceed validation.full_max_bytes.')
            if done == total or done % 10 == 0:
                self.update(info=dict(type='full_validation', status='progress', epoch=epoch,
                                      case=case, completed=done, total=total, unit='tiles'))
        self.check_cancel()
        return self.ref._instances_from_prediction(shape, np.concatenate(probabilities), np.concatenate(distances),
            points=np.concatenate(points), prob_class=np.concatenate(classes) if classes else None)[0]

    def optimize_thresholds(self):
        """Keep 2D threshold optimization, but never collect an unbounded dataset."""
        images, labels, size = [], [], 0
        budget = int(self.options.get('threshold_max_bytes', self.preview_bytes))
        per_pixel = (self.ref.config.n_channel_in * 4 + 8
                     + 8 * (self.ref.config.n_rays + 1) / math.prod(self.ref.config.grid))
        if sum(math.prod(y.shape) * per_pixel for y in self.labels) <= budget:
            return self.ref.optimize_thresholds([np.asarray(x) for x in self.images],
                                               [np.asarray(y) for y in self.labels], save_to_json=True)
        for first in range(self.sample_count):
            self.check_cancel()
            estimate = math.prod(self.patch) * per_pixel
            if size + estimate > budget:
                break
            inputs, _, masks = self._batch(first, first + 1)
            images.append(inputs[0][0])
            labels.append(masks[0])
            size += estimate
        if not images:
            raise MemoryError('No threshold-optimization patch fits validation.threshold_max_bytes.')
        self.update(message='Optimizing thresholds on %d fixed validation patches (bounded working set).' % len(images),
                    info={'type': 'thresholds', 'samples': len(images)})
        return self.ref.optimize_thresholds(images, labels, save_to_json=True)


class ValidationArtifacts(Callback):
    def __init__(self, validation):
        super().__init__()
        self.validation = validation
        self.checkpoint = next((cb for cb in validation.ref.callbacks
                                if getattr(cb, 'save_best_only', False)
                                and getattr(cb, 'monitor', None) == 'val_loss'), None)
        self.best = getattr(self.checkpoint, 'best', None)
        self.best_path = Path(self.checkpoint.filepath) if self.checkpoint is not None else None
        self.best_exists = self.best_path is not None and self.best_path.is_file()

    def on_epoch_end(self, epoch, logs=None):
        # Runs after checkpoint callbacks, so a diagnostic failure cannot lose the best model.
        if self.checkpoint is not None:
            best = self.checkpoint.best
            if best is not None and (self.best is None or best < self.best) and self.best_path.is_file():
                self.validation.update(
                    message='New best validation model at epoch %d: val_loss=%.5f. %s: %s' % (
                        epoch + 1, best, 'Overwritten checkpoint' if self.best_exists else 'Saved at', self.best_path),
                    info=dict(type='checkpoint', kind='best', status='saved', epoch=epoch + 1,
                              path=str(self.best_path), overwrite=self.best_exists,
                              monitor='val_loss', value=float(best)))
                self.best_exists = True
            self.best = best
        last_name = self.validation.ref.config.train_checkpoint_last
        if last_name:
            path = self.validation.output / last_name
            overwrite = path.exists()
            temporary = path.with_name(path.stem + '.tmp.h5')
            self.model.save_weights(str(temporary))
            os.replace(temporary, path)
            self.validation.update(message=('Overwriting' if overwrite else 'Saving') + ' StarDist last checkpoint: ' + str(path),
                                   info=dict(type='checkpoint', kind='last', status='saved', epoch=epoch + 1,
                                             path=str(path), overwrite=overwrite))
        tf.train.Checkpoint(model=self.model, optimizer=self.model.optimizer).write(
            str(self.validation.output / ('training_state-%06d' % (epoch + 1))))
        self.validation.save_state(epoch + 1)
        self.validation.finish_epoch(epoch + 1)


def train_with_validation(model, images, labels, validation, epochs, steps):
    axes = model.config.axes.replace('C', '')
    border = model.config.train_completion_crop if model.config.n_dim == 2 and model.config.train_shape_completion else 0
    if any((p - 2 * border) % d for p, d in zip(model.config.train_patch_size, model._axes_div_by(axes))):
        raise ValueError('StarDist training patch shape is incompatible with the model downsampling.')
    rng = np.random.RandomState(int(validation.options.get('seed', 42)))
    if validation.initial_epoch:
        tf.train.Checkpoint(model=model.keras_model, optimizer=model.keras_model.optimizer).restore(
            str(validation.output / validation.resume_checkpoint)).expect_partial()
        best = min((entry['losses']['loss'] for entry in validation.history), default=float('inf'))
        for callback in model.callbacks:
            if hasattr(callback, 'best') and getattr(callback, 'monitor', None) == 'val_loss':
                callback.best = best
    def batches():
        for _ in range(max(0, epochs - validation.initial_epoch) * steps):
            with validation.sampling_lock:
                validation.check_cancel()
                plan = []
                for _ in range(model.config.train_batch_size):
                    i = int(rng.randint(len(images)))
                    plan.append((i, sample_start(labels[i], model.config.train_patch_size, rng,
                                                 rng.rand() < model.config.train_foreground_only)))
                inputs, targets, _ = crop_batch(model, images, labels, plan)
            yield inputs, targets
    callbacks = [validation] + model.callbacks + [ValidationArtifacts(validation)]
    try:
        worker_kwargs = {} if IS_KERAS_3_PLUS else dict(workers=0, use_multiprocessing=False)
        history = model.keras_model.fit(batches(), epochs=epochs, steps_per_epoch=steps,
                                        initial_epoch=validation.initial_epoch, callbacks=callbacks, verbose=0, **worker_kwargs)
    except ValidationCancelled:
        return None
    finally:
        validation.schedule.close()
    model._training_finished()
    return history
