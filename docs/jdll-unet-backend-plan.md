# JDLL UNet Backend Plan

This document defines the first JDLL-owned UNet backend to be driven from Appose. It is intended as an implementation handoff for an agent starting from scratch.

The objective is a lightweight, configurable UNet training and inference package that works reliably from the JDLL Java plugin on Linux, macOS, and Windows, while hiding most deep-learning complexity from bioimage analysts.

## Scope

Implement a Python package named `jdll_unet`.

The first implementation targets standard UNet-style models. Do not implement nnU-Net in this phase. The future nnU-Net integration should be separate and should reuse only compatible dataset/task metadata.

The Java UI should expose only simple decisions:

- Model name.
- Dataset path.
- Starting point: train from scratch or fine tune an existing UNet model.
- Model size: tiny or medium. Pro/nnU-Net is reserved for later.
- Number of epochs.
- Acceleration checkbox, resolved to CPU, CUDA, or MPS where supported.

The backend should infer the actual segmentation task and training target from the dataset whenever possible.

## Non-Goals

Do not add a heavy framework dependency such as MONAI, PyTorch Lightning, BiaPy, or segmentation-models-pytorch for the first backend.

Do not expose augmentation, loss, optimizer, postprocessing, or architecture details in the first UI iteration.

Do not implement full nnU-Net planning, folds, cascades, ensembling, or spacing-aware medical image resampling in this phase.

Do not rely on cooperative cancellation for correctness. JDLL may close/kill the Appose Python process on cancel. The trainer must save useful checkpoints continuously enough that process termination is acceptable.

## Dependency Budget

Keep dependencies small and cross-platform.

Required:

- `python >= 3.10`
- `torch`
- `numpy`
- `tifffile` or `imageio`

Recommended:

- `scipy`

Avoid unless there is a strong reason:

- `scikit-image`
- `opencv`
- `albumentations`
- `MONAI`
- `lightning`

Use PyTorch tensor ops and `torch.nn.functional.grid_sample` for affine/elastic-like augmentation where practical. Use `scipy` for connected components, distance transforms, and simple morphology if needed.

## Package Layout

Recommended package structure:

```text
jdll_unet/
  __init__.py
  config.py
  schema.py
  errors.py
  io.py
  model.py
  dataset.py
  task_detect.py
  targets.py
  augment.py
  losses.py
  metrics.py
  trainer.py
  infer.py
  postprocess.py
  appose_api.py
```

## Maintainability Requirements

Implement `jdll_unet` as a small Python library first and an Appose integration second.

The public Python API must be stable, typed, and usable without Java:

```python
from jdll_unet import detect_task, infer, load_model, train
from jdll_unet.config import InferenceConfig, TrainConfig

result = train(TrainConfig(...))
prediction = infer(InferenceConfig(...), image)
```

`appose_api.py` must be a thin boundary layer only. It should parse JSON/dicts, call the public API, and serialize outputs. It must not contain model, dataset, augmentation, training, inference, or postprocessing logic.

Do not pass raw dictionaries through the internal package. Raw dicts are allowed only at boundaries:

- Java/Appose input.
- CLI input if a CLI is added later.
- JSON config loading.
- JSON serialization output.

After parsing, use typed dataclasses or equivalent typed config objects.

Required typed config objects:

- `TrainConfig`
- `InferenceConfig`
- `ModelConfig`
- `DatasetConfig`
- `TaskConfig`
- `AugmentationConfig`
- `LossConfig`
- `OptimizerConfig`
- `PostprocessConfig`
- `TilingConfig`
- `PreviewConfig`

These objects must:

- Have defaults.
- Validate themselves or be validated by `schema.py`.
- Be serializable to JSON.
- Be deserializable from JSON.
- Preserve unknown future-compatible fields when possible, or fail with a clear schema-version error.

Use small composable classes instead of one large trainer script:

- `DatasetIndex`: discovers and pairs data.
- `TaskDetector`: infers task and ambiguity.
- `TargetBuilder`: converts masks into training targets.
- `AugmentationPipeline`: applies the selected augmentation preset.
- `UNetFactory`: creates models from `ModelConfig`.
- `CheckpointManager`: saves best/last/model checkpoints atomically.
- `ProgressReporter`: sends JSON-safe progress to Appose or logs.
- `PreviewWriter`: writes validation previews.
- `Trainer`: owns the training loop and composes the objects above.
- `InferenceRunner`: owns tiled inference and postprocessing.

Use clear exceptions from `errors.py`:

- `JdllUnetError`
- `ConfigError`
- `DatasetError`
- `TaskDetectionError`
- `ModelFormatError`
- `TrainingError`
- `InferenceError`

Error messages must be actionable for a GUI user. Avoid raw stack traces for expected validation failures.

## Public API Contract

`__init__.py` must expose only stable public entry points:

```python
def detect_task(config: TrainConfig | dict) -> TaskDetectionResult:
    ...

def train(config: TrainConfig | dict, progress=None) -> TrainResult:
    ...

def load_model(model_path: str | Path, device: str = "cpu") -> LoadedUnetModel:
    ...

def infer(config: InferenceConfig | dict, image, model=None, progress=None) -> InferenceResult:
    ...
```

Public result objects must also be typed and JSON-serializable:

- `TaskDetectionResult`
- `TrainResult`
- `InferenceResult`
- `ModelInfo`

The direct Python API must not depend on Appose. `progress` should be a generic callable that receives JSON-serializable dicts. `appose_api.py` adapts this callable to `task.update`.

The package should remain importable in environments without Java, Icy, Fiji, or ImageJ.

## Configuration and Versioning

The package must use versioned schemas.

Recommended constants:

```python
PACKAGE_FORMAT = "jdll-unet"
CONFIG_VERSION = 1
MODEL_FORMAT_VERSION = 1
```

Every saved model must include:

- Package format.
- Model format version.
- Training config version.
- Package version if available.
- Torch version.
- Creation timestamp.

Model loading must validate format/version before loading weights. If a model is too new, fail with a clear `ModelFormatError`.

Atomic write requirements:

- Write JSON to a temporary file, then `os.replace`.
- Write checkpoints to temporary files, then `os.replace`.
- Never leave a half-written `model.pt`, `weights_best.pt`, `weights_last.pt`, or `config.json`.

### `config.py`

Parse and validate training/inference configuration.

All important behavior must be configurable by JSON/dict even if the Java UI does not expose it yet.

`config.py` should define the typed dataclasses and conversion helpers:

```python
TrainConfig.from_dict(...)
TrainConfig.to_dict()
TrainConfig.from_json_file(...)
TrainConfig.to_json_file(...)
```

Validation should be deterministic and independent of the training loop.

Resolve `"auto"` values in one explicit step, e.g.:

```python
resolved = resolve_train_config(config, dataset_summary, task_result)
```

Do not silently mutate the original config object. Return a resolved copy.

Required top-level training config fields:

```json
{
  "model_name": "my_unet",
  "output_dir": "/path/to/models/unet/my_unet",
  "dataset_path": "/path/to/dataset",
  "starting_point": "scratch",
  "base_model": null,
  "architecture": "tiny-2d",
  "device": "cpu",
  "epochs": 100,
  "seed": 42
}
```

Important optional fields:

```json
{
  "task": "auto",
  "axes": "auto",
  "input_channels": "auto",
  "output_classes": "auto",
  "patch_size": "auto",
  "batch_size": "auto",
  "learning_rate": "auto",
  "optimizer": "adamw",
  "weight_decay": 0.00001,
  "validation_fraction": 0.15,
  "foreground_oversampling": "auto",
  "augmentation_profile": "auto",
  "num_workers": 0,
  "mixed_precision": "auto",
  "save_every_epoch": true,
  "preview_count": 20
}
```

Use conservative defaults. Never require the Java side to specify expert parameters.

### `schema.py`

Own schema validation and migration.

Responsibilities:

- Validate user-provided config dicts before dataclass construction.
- Validate saved model `config.json`.
- Migrate older config versions if migration is trivial and safe.
- Fail loudly when a future version cannot be loaded safely.

Keep schema logic separate from training/inference code.

### `errors.py`

Define package-specific exceptions.

All expected user/data/config problems should raise `JdllUnetError` subclasses. Appose and Java wrappers should catch these errors and show concise messages.

### `io.py`

Load images and masks from disk.

Supported image formats:

- TIFF/TIF, including 16-bit masks.
- PNG.
- JPEG only for raw images, not preferred for masks.

Supported dataset layouts:

```text
dataset/
  images/
  masks/
```

```text
dataset/
  train/
    images/
    masks/
  val/
    images/
    masks/
```

Accept common folder aliases:

- Images: `images`, `image`, `imgs`, `img`, `data`.
- Masks: `masks`, `mask`, `labels`, `label`, `gt`.

Pair images and masks by stem. Accept suffix variants:

- `image001.tif` with `image001.tif`
- `image001.tif` with `image001_mask.tif`
- `image001_image.tif` with `image001_label.tif`

Mask loading requirements:

- Preserve integer values.
- Support 8-bit, 16-bit, 32-bit masks.
- Treat `0` as background.
- Do not normalize masks.
- Collapse mask RGB only if it clearly represents a single label plane duplicated across channels. Otherwise fail with a clear error.

Image loading requirements:

- Convert image tensors to `float32`.
- Preserve channel dimension when present.
- Support grayscale, multichannel 2D, and stacks for 2.5D.
- Normalize according to config.

`io.py` must not decide the segmentation task. It only loads data and returns typed metadata such as shape, dtype, axes, and label value summaries.

Keep all image arrays in a consistent internal axis convention:

- 2D image: `C, Y, X`
- 2D mask: `Y, X`
- 2.5D sampled image: `C, Y, X`, where neighboring slices are folded into channels
- Future 3D image: `C, Z, Y, X`
- Future 3D mask: `Z, Y, X`

Document every boundary where axes are converted.

### `dataset.py`

Create PyTorch datasets and samplers.

Responsibilities:

- Build train/validation indices from `DatasetIndex`.
- Sample patches.
- Apply foreground oversampling.
- Apply 2.5D context extraction.
- Call `TargetBuilder`.
- Call `AugmentationPipeline`.
- Return PyTorch tensors in the model's expected axis order.

Do not load the entire dataset into GPU memory.

Keep CPU memory usage bounded. It is acceptable to cache small metadata and normalized file-level arrays only when explicitly configured.

Dataset sampling must be deterministic when `seed` is set.

### `task_detect.py`

Infer task type using metadata first, then mask statistics.

Supported tasks:

- `binary_semantic`
- `multiclass_semantic`
- `instance_friendly`

Metadata signals:

| Signal | Preferred task |
| --- | --- |
| Binary mask | Binary semantic |
| Class mask with named labels | Multiclass semantic |
| Fiji/Icy ROI manager one ROI per object | Instance-friendly |
| Label image with unique object IDs | Instance-friendly |
| Separate class channels | Semantic or multilabel semantic |
| Bounding boxes | Route away from UNet to YOLO |
| Points | Route away from UNet or future detection workflow |

Compute these per-mask statistics:

- Number of unique nonzero labels.
- Connected components per label value.
- Consistency of label values across images.
- Whether class names exist.
- Whether labels are sequential object IDs.
- Whether many disconnected components share one label value.

Scoring:

```python
instance_score = 0

if annotation_source == "roi_manager_one_roi_per_object":
    instance_score += 4

if median_unique_labels_per_image > 10:
    instance_score += 3

if most_label_values_have_one_connected_component:
    instance_score += 3

if label_values_are_not_consistent_across_images:
    instance_score += 2

if class_names_exist:
    instance_score -= 4

if unique_label_set_is_small_and_stable:
    instance_score -= 3

if many_components_share_the_same_label_value:
    instance_score -= 2
```

Decision:

- `score >= 4`: `instance_friendly`
- `score <= -2`: semantic
- otherwise: ambiguous

For ambiguous datasets, return a result that lets Java ask:

> Do different numbers in the annotation represent different biological classes or different individual objects?

Choices:

- Different classes.
- Different objects.

Do not ask users whether they want "semantic" or "instance" unless an advanced mode exists later.

Return a typed result:

```python
TaskDetectionResult(
    task="instance_friendly",
    confidence="high",
    score=6,
    ambiguous=False,
    question=None,
    summary={...}
)
```

The summary should be JSON-serializable and useful for debugging/logging.

### `targets.py`

Convert masks to learning targets.

Binary semantic:

- Input mask values: `0` background, nonzero foreground.
- Target: foreground binary mask.

Multiclass semantic:

- Input mask values: `0..C`.
- Target: class index mask.
- Optional future support: one-hot/multilabel masks.

Instance-friendly:

- Input mask values: `0` background, `1..N` arbitrary object IDs.
- Targets:
  - Foreground binary mask.
  - Boundary or separation map.
  - Optional distance/regression map.

Instance-friendly first implementation should start with foreground + boundary/separation. Add distance maps only if the postprocessing needs it.

### `model.py`

Implement JDLL-owned UNet variants.

Model code must be independent from training and inference.

Required factory:

```python
model = UNetFactory.create(ModelConfig(...))
```

Do not instantiate models by scattered conditional logic in the trainer.

Tiny 2D:

- Small enough for CPU training.
- Suggested defaults:
  - Base channels: 16.
  - Depth: 3.
  - Conv blocks per level: 2.
  - Normalization: batch norm or instance norm configurable.
  - Dropout: disabled by default or very low.

Medium 2D:

- Comfortable on common laptop GPUs.
- Still feasible on CPU for long runs.
- Suggested defaults:
  - Base channels: 32.
  - Depth: 4.
  - Conv blocks per level: 2.
  - Normalization configurable.
  - Optional dropout.

2.5D:

- Use the same 2D model.
- Feed neighboring slices as input channels.
- Predict the center slice.
- Configurable context depth, default `3` slices.

3D:

- Not required in the first backend milestone.
- Plan for medium 3D after 2D/2.5D is stable.
- Avoid tiny 3D initially unless there is a strong use case.

Architecture config must support:

- Input channels.
- Output channels/classes.
- Base channels.
- Depth.
- Conv blocks per level.
- Normalization type.
- Activation.
- Dropout.
- 2D vs 2.5D vs future 3D.

Checkpoint state should save:

- Model state dict.
- Model config.
- Task config.
- Normalization/postprocessing config.
- Epoch.
- Best metric.
- Optimizer state for `weights_last.pt`.

`model.pt` may omit optimizer state and should be optimized for inference loading.

### `augment.py`

Implement augmentation profiles inspired by nnU-Net, but cheaper.

All augmentation parameters must be configurable.

Augmentations must apply consistently to image and mask/targets for spatial transforms. Intensity transforms apply only to images.

Always keep:

- Random patch/crop sampling.
- Foreground oversampling.
- Flips along valid spatial axes.
- 90-degree rotations for 2D when shape allows it.
- Intensity normalization.
- Random brightness/contrast.
- Gamma augmentation.
- Gaussian noise.
- Mild Gaussian blur.
- Random channel dropout for multichannel inputs.

Balanced GPU or optional CPU:

- Small affine rotation, e.g. `[-15, 15]` degrees.
- Small scaling, e.g. `[0.85, 1.25]`.
- Random anisotropic scaling in 2D.
- Simulated low-resolution augmentation.
- Boundary/separation target generation for instance-label masks.

Strong mode only:

- Elastic deformation.
- Large rotations.
- Heavy scaling.
- Strong blur.
- Strong low-resolution simulation.
- 3D spatial transforms.

Default augmentation profile selection:

```text
Tiny CPU/GPU: fast
Medium CPU: fast or light-balanced
Medium CUDA/MPS: balanced
Future Pro/nnU-Net: strong
```

Foreground oversampling must be enabled for both Tiny and Medium.

Suggested defaults:

```text
Tiny:
  foreground_oversampling: true
  foreground_probability: 0.33 to 0.5
  augmentation_profile: fast

Medium:
  foreground_oversampling: true
  foreground_probability: 0.5
  augmentation_profile: balanced on GPU/MPS
  augmentation_profile: fast or light-balanced on CPU

Instance-friendly:
  foreground/object_oversampling: true
  foreground_probability: 0.5 to 0.67
```

If no foreground crop can be found, fall back to random crops.

Implementation constraints:

- Spatial transforms must use nearest-neighbor interpolation for masks/labels.
- Spatial transforms must use bilinear interpolation for images in 2D.
- Intensity transforms must never be applied to masks.
- Random state must be seedable.
- The same random spatial transform must be applied to image and all target maps.

Keep augmentation presets as data/config, not hardcoded branches. Example:

```python
AUGMENTATION_PRESETS = {
    "fast": AugmentationConfig(...),
    "balanced": AugmentationConfig(...),
    "strong": AugmentationConfig(...),
}
```

The selected preset can be overridden by explicit config fields.

### `losses.py`

The user must not choose the loss in the first UI.

Automatic losses:

| Task | Loss |
| --- | --- |
| Binary semantic | Dice + binary cross entropy |
| Multiclass semantic | Dice + cross entropy |
| Instance-friendly foreground | Dice + binary cross entropy |
| Boundary/separation map | BCE or focal loss |
| Distance/regression map | L1, MSE, or smooth L1 |

Dice-based losses are required because foreground objects are often sparse.

Loss weights must be configurable in JSON.

### `metrics.py`

Minimum training/validation metrics:

| Task | Metrics |
| --- | --- |
| Binary semantic | Dice, IoU |
| Multiclass semantic | mean Dice, per-class Dice |
| Instance-friendly | foreground Dice, boundary loss, optional object count estimate |

Metrics must be JSON-serializable so they can be sent through `task.update`.

### `trainer.py`

Implement a simple PyTorch training loop.

Recommended internal object graph:

```python
trainer = Trainer(
    config=resolved_config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    losses=loss_bundle,
    metrics=metric_bundle,
    checkpoint_manager=checkpoint_manager,
    progress_reporter=progress_reporter,
    preview_writer=preview_writer,
)
```

The trainer should be readable and boring. Avoid callback frameworks unless they are tiny package-local callables.

Requirements:

- Load dataset and infer task before training.
- Create train/validation split if no validation folder exists.
- Warn through `task.update` if very few images exist.
- Save best validation checkpoint.
- Save last checkpoint.
- Save training metrics.
- Generate validation previews.
- Emit Appose progress updates.
- Write logs to `training.log`.
- Support CPU, CUDA, and MPS device strings.
- Use `num_workers=0` by default for Appose stability.
- Mixed precision only on CUDA by default.

Progress updates should include:

```json
{
  "type": "progress",
  "epoch": 1,
  "step": 20,
  "total_epochs": 100,
  "total_steps": 1000,
  "losses": {
    "train/total_loss": 0.42,
    "train/dice_loss": 0.2
  },
  "metrics": {
    "val/dice": 0.78
  }
}
```

Checkpoint behavior:

- Save `weights_last.pt` at the end of every epoch.
- Save `weights_best.pt` whenever validation improves.
- Keep `model.pt` as a copy or alias of the best checkpoint after training completes.
- If the process is killed, `weights_last.pt` and possibly `weights_best.pt` should already exist from the last completed epoch.

Cancellation behavior:

- Java may kill the Python process.
- The trainer does not need complex cooperative cancellation.
- The trainer should checkpoint often enough to make hard cancellation acceptable.

Logging:

- All logs go to `training.log`.
- Progress events go through `ProgressReporter`.
- Expected user errors should be concise.
- Debug stack traces can be logged to file but should not be the primary Java message.

Reproducibility:

- Set Python, NumPy, and Torch seeds.
- Store the seed in `config.json`.
- Store the resolved config, not only the user config.
- Deterministic CUDA is optional and should default to false because it may slow training.

### `infer.py`

Load a trained UNet model folder and run inference.

Inference must use `LoadedUnetModel` or an equivalent typed wrapper that contains:

- Model.
- Device.
- Model config.
- Task config.
- Normalization config.
- Postprocessing config.
- Model path.

Do not repeatedly parse config or recreate the network inside every tile.

Inputs:

- Model folder or checkpoint path.
- Image tensor/array.
- Device.
- Optional tiling config.

Outputs by task:

Binary semantic:

- Foreground probability map.
- Binary mask.
- Optional connected-component labels.

Multiclass semantic:

- One probability map per class.
- Final class-label image via argmax.

Instance-friendly:

- Foreground probability map.
- Boundary/separation map.
- Final labeled instance image.

Inference must support tiled prediction for large images. Tile overlap and blending must be configurable.

Tiling requirements:

- Tile size defaults to training patch size.
- Overlap defaults to 25 percent.
- Blend overlapping tile probabilities with a smooth weight map.
- Preserve final output size.
- Support cancellation/progress events between tiles.

### `postprocess.py`

Postprocessing should depend on task.

Binary semantic:

- Threshold probability map.
- Remove tiny objects.
- Fill small holes optionally.
- Optional connected components for labels.

Multiclass semantic:

- Argmax over class probabilities.
- Optional class-wise cleanup.

Instance-friendly:

- Threshold foreground.
- Use boundary/separation prediction.
- Connected components first.
- Watershed-style postprocessing later if needed.
- Remove small objects.
- Relabel objects.

UI should expose no postprocessing controls in the first iteration. Defaults must be stored in `config.json`.

### `appose_api.py`

Expose functions that Java-generated Appose scripts can call.

Recommended minimal API:

```python
def train(config: dict, task=None) -> dict:
    ...

def infer(config: dict, inputs: dict, task=None) -> dict:
    ...

def detect_task(config: dict) -> dict:
    ...
```

Return JSON-serializable metadata and shared-memory output descriptors as needed by existing JDLL Appose conventions.

Avoid global mutable state except for loaded inference model caching inside one Appose process. Training should start a fresh process every time.

Implementation rules:

- Convert inbound dicts to typed configs immediately.
- Catch `JdllUnetError` and return concise expected failures.
- Let unexpected exceptions propagate after logging.
- Never import Java/Icy/Fiji-specific code.
- Keep shared-memory details isolated here or in a tiny `shm.py` helper if needed.

Optional CLI:

The package may add `python -m jdll_unet train config.json` and `python -m jdll_unet infer config.json`, but CLI support must reuse the public API and must not duplicate Appose logic.

## Model Folder Format

Every trained model should live in:

```text
models/unet/<model_name>/
```

Required files:

```text
config.json
weights_best.pt
weights_last.pt
model.pt
training.log
metrics.json
```

Optional files:

```text
previews/latest.json
previews/epoch_0001.json
previews/preview_000_image.npy
previews/preview_000_prediction.npy
```

`config.json` must contain enough information to run inference without Java knowing the architecture details.

Required `config.json` fields:

```json
{
  "format": "jdll-unet",
  "format_version": 1,
  "task": "binary_semantic",
  "architecture": "tiny-2d",
  "input_axes": "yx",
  "output_axes": "yx",
  "input_channels": 1,
  "num_classes": 1,
  "normalization": {
    "type": "percentile",
    "low": 1.0,
    "high": 99.8
  },
  "postprocessing": {
    "threshold": 0.5,
    "min_object_size": 0
  }
}
```

Also include a `resolved_training` section with all auto parameters resolved:

```json
{
  "resolved_training": {
    "patch_size": [256, 256],
    "batch_size": 8,
    "learning_rate": 0.0003,
    "augmentation_profile": "fast",
    "foreground_probability": 0.5
  }
}
```

Backward compatibility:

- New package versions must continue loading `MODEL_FORMAT_VERSION = 1`.
- If compatibility must break, increment `MODEL_FORMAT_VERSION` and add a migration note.
- Do not change meanings of existing config fields silently.

## Java Integration Plan

Initial Java classes should mirror the StarDist/YOLO pattern:

- `UnetTrainingConfig`
- `UnetTrainingService`
- `UnetInferenceService`
- `UnetInstaller`
- `UnetModelRegistry`

Training:

- Start a fresh Appose Python process for each training run.
- Send config as JSON.
- Display progress from `task.update`.
- Close/kill the process on cancel.
- Refresh `models/unet` registry after training completes.

Inference:

- Keep one model loaded per process if model path and device do not change.
- Unload and close process when model path or device changes.
- Cancel inference without unloading the loaded model where possible, matching StarDist/YOLO behavior.

Task detection:

- Java should be able to call `detect_task` before actual training.
- If the result is ambiguous, Java asks the simple classes-vs-objects question and passes the answer back in training config.

## Implementation Milestones

### Milestone 1: Binary 2D Tiny

- Package skeleton.
- 2D Tiny UNet.
- Image/mask pairing.
- Binary semantic task.
- Dice + BCE.
- Fast augmentation.
- Foreground oversampling.
- Checkpoints and progress updates.
- Inference with thresholded mask.

### Milestone 2: Multiclass 2D

- Multiclass task detection.
- Class-index targets.
- Dice + CE.
- Per-class metrics.
- Argmax output.

### Milestone 3: Medium 2D and 2.5D

- Medium architecture.
- 2.5D input sampling.
- Balanced augmentation on CUDA/MPS.
- Fine tuning from existing UNet model folders.

### Milestone 4: Instance-Friendly

- Instance-label task detection.
- Foreground + boundary/separation targets.
- Instance postprocessing.
- Object-label output.

### Milestone 5: Java Service Integration

- Wire training/inference services from the UNet UI.
- Model folder refresh.
- Cancellation via process close.
- Shared-memory output handling.

### Milestone 6: 3D and nnU-Net Preparation

- Add medium 3D only if the 2D/2.5D backend is stable.
- Keep nnU-Net as separate pro backend.

## Testing Requirements

Python tests:

- Config dataclass defaults, validation, serialization, and migration.
- Dataset pairing for common folder layouts.
- Task detection on binary, multiclass, and instance-label masks.
- Target generation.
- Augmentation shape/dtype consistency.
- Augmentation reproducibility with a fixed seed.
- UNet factory output shapes for Tiny and Medium.
- Loss functions on small tensors.
- Checkpoint save/load round trip.
- Tiny training smoke test on CPU with a synthetic dataset.
- Inference smoke test from saved `model.pt`.
- Appose API boundary test using plain dicts and a fake progress object.

Java tests or manual checks:

- UI validation accepts `models/unet/<model>/model.pt`.
- Training button sends expected config.
- Cancel closes the Appose process.
- Inference reloads when model or device changes.
- Output masks have expected axes and dtype.

Quality gates:

- A synthetic CPU training smoke test should finish in under one minute on a typical laptop.
- Importing `jdll_unet` must not import Appose or Java-specific dependencies.
- Importing `jdll_unet` must not initialize CUDA.
- `load_model(..., device="cpu")` must work even if the model was trained on CUDA.

## Design Principles

Keep the backend small and inspectable.

Own the training loop so JDLL controls Appose progress, cancellation, checkpoints, and model export.

Make defaults strong enough for bioimage users, but keep every expert parameter configurable through JSON for future UI or scripting use.

Prefer cheap, high-value nnU-Net-inspired behavior first: foreground oversampling, normalization, Dice losses, validation checkpoints, and balanced augmentations.

Prefer explicit boundaries over convenience shortcuts. The package should remain useful as:

- A JDLL/Appose backend.
- A direct Python library for tests and debugging.
- A future CLI tool.
- A future foundation for nnU-Net-style pro workflows without entangling the simple UNet backend.
