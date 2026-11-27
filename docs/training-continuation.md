# Continuing a DeepIcy training run

Select the interrupted model under **Fine-tune**. The model picker shows the checkpoint and epoch it will use. For example, selecting epoch 38 of 100 suggests 62 additional epochs. You can change that number. Training creates a separate model directory. A completed model opens with the workflow's usual fine-tuning settings.

DeepIcy first looks for the best validated checkpoint with its optimizer, then the last validated checkpoint with its optimizer. If only weights survive, it prefers best weights and initializes a new optimizer. Native validation still determines what “best” means for each workflow. For example, StarDist selects by validation loss and Cross-GOOSE by AP at IoU 0.75.

An unchanged training budget and update count allow the backend to restore its learning-rate trajectory. When either changes, the trainer keeps the current optimizer and adapts future learning rates to the new budget. A change to the network's input or output layers uses the workflow's weight-transfer and fresh-optimizer path. The training log reports the choice.

## Validation data

An automatic split produces `split.json` in the model directory. It records original image and annotation names relative to the dataset root. A spatial holdout also records the pixel regions. User-supplied train and validation folders take precedence.

On continuation, surviving sources keep their roles and new sources are divided between training and validation. One remaining annotated validation source is sufficient. If no held-out annotations remain, the preparer moves an annotated source into validation or uses a supported disjoint spatial holdout. A changed validation set is scored before the first optimizer update. The child saves that baseline as its best checkpoint until a later epoch beats it.

## Run files

`continuation.json` records the cumulative epoch budget, run status, and the validated epoch associated with each checkpoint. Each backend keeps its native inference weights and stores the training state required by its optimizer alongside them. When a backend writes weights and optimizer state separately, the run also retains the indexed epoch's weights under `.continuation/`. This lets a child recover a matching pair if the parent process exits while publishing the next epoch. Retain the model directory to continue training. An exported weights file can still be used for ordinary fine-tuning.

Runs written by older versions are recognized when their configuration, validation history and checkpoints establish an interrupted run. Older formats vary in what they saved. For example, earlier DeepIcy Cellpose runs can recover segmentation weights, while an interrupted run produced by the new trainer can also recover AdamW's moments.
