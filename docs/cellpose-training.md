# Cellpose training

DeepIcy's Cellpose training tab offers fine-tuning and scratch training. Both use Cellpose 3's native 2D CPnet. For fine-tuning, select cyto3, cyto2, cyto, nuclei, native weights, or an earlier run directory. Scratch training uses cyto3 capacity with fresh weights. Cellpose starts at zero learning rate, so a fresh run needs at least two epochs. A continuation can run for one additional epoch.

## Dataset and channels

Place paired fields in `images/` and `masks/`. For an explicit holdout, use `train/images/`, `train/masks/`, `val/images/`, and `val/masks/`. The shared JDLL segmentation dataset preparer also recognizes mixed image/mask folders and split-folder aliases. JDLL-U-Net readers validate image formats, axes and label values. Masks contain nonnegative integer instance IDs with zero background. In each XY plane, disconnected regions sharing an ID become separate objects. Touching objects need separate IDs. Cellpose prepares flow targets once and applies its native flow-aware crop, rotation and scale augmentation during training.

The first channel selects cell intensity and the second optionally selects nuclear intensity. Indices start at one. Zero takes the channel mean for cell intensity and leaves the nuclear input empty. Training and inference apply the same mapping. Z stacks contribute labeled XY planes. Generated splits hold out entire annotated sources. A single annotated source needs a spatial split with complete cells on both sides. Empty training planes are skipped and empty validation planes contribute false positives to object scoring.

## Monitoring and saved models

Cellpose calls the JDLL adapter after optimizer updates and after the epoch's validation batches. The UI samples batch-loss updates on large datasets. Native validation loss covers all validation fields. Panoptic quality uses complete decoded masks on up to eight fixed validation planes. **Full validation** scores all validation planes at the next epoch boundary. Preview assets retain the image, instance labels and prediction independently so their contours can be inspected in the viewer.

Runs are stored under `models/cellpose/<name>/`. Reusing a name creates a new directory. Open the run directory for inference, fine-tuning or continuation. When the validation set changes on continuation, Cellpose measures the starting checkpoint's panoptic quality before updating the network. A subsequent epoch must improve that value to replace the baseline. Between trained epochs, validation loss breaks ties in panoptic quality.

Before writing the temporary training TIFFs, JDLL estimates the space required on the run directory's filesystem for staged fields, checkpoints, atomic replacements, logs and metadata. Insufficient required space stops training and reports the path, estimated GB required and GB free in the training log. Preview images have a separate budget. If only previews exceed the available space, validation metrics and training continue and the preview panel explains the shortage.

| File | Contents |
| --- | --- |
| `best.pt` | Native segmentation weights selected by highest held-out panoptic quality, then lowest validation loss |
| `last.pt` | Weights from the last completed validation epoch |
| `best.pt_size.npy` | Native diameter regressor fitted on the training fields using the selected best network |
| `model.json` | Format version, backend identifier, selected checkpoint, input channels, initialization provenance, best epoch and diameters in pixels |
| `config.json`, `dataset.json` | Requested settings and the source fields, regions and plane indices used in each split |
| `run.json`, `metrics.json` | Completion state, execution settings and epoch history |
| `previews/`, `training.log` | Validation examples and training messages |

Both checkpoints use the ordinary Cellpose state dictionary. Fine-tuning starts a new optimizer. The reference diameter is 30 px for cyto3 scratch training. Fine-tuning keeps the base network's reference scale, including 17 px for legacy nuclei. Cellpose measures the median equivalent-circle diameter on each training plane, floors it at 5 px, and stores the mean of those measurements as a fallback. Completed training fits the diameter regressor to the best checkpoint.

With automatic diameter and a matched regressor, DeepIcy extracts style from the original XY plane, predicts diameter, and segments at that diameter. A supplied diameter skips the style pass. Older checkpoints without size weights use their stored training diameter. Selecting **Display intermediate outputs** also denoises the selected channels at the resolved diameter before segmentation, and shows the restored image and flows. Enabling this option can change the labels. Cellpose downloads its denoising weights on first use.

## Backend contract

Java callers construct a `CellposeTrainingRequest` and call `CellposeTrainingService.train`. A null base requests scratch training. The service supports cancellation and full-validation requests on a single run. Its result is the model directory. The manifest's version and `cellpose3-cpnet` backend identify the checkpoint format independently of its filename and training origin.

The shared PyTorch Pixi manifests pin the Cellpose 3.1.1.1 fork by commit. It adds `on_train_begin`, `start_epoch`, `on_step`, `on_validation_end` and `validate_each_epoch` and corrects diameter measurement when every pixel has an instance label. [Fork patch](https://github.com/carlosuc3m/cellpose/compare/fb22843e70d03f7884c301b7b72bedd7d9c3d2d9...c1c118080909c17d3ed05b60260c39b666e6ce4a)
