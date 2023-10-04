import xarray as xr
from stardist.models import StarDist2D

import os
import shutil
import os
from pathlib import Path


def assertions(model_path, images, ground_truth, new_model_dir):

  assert isinstance(model_path, str), "The input argument 'model_path' must be a string, either the name" \
  + " of one of the default pre-trained Stardist models or the directory to a pre-trained Stardist model"

  assert isinstance(new_model_dir, str), "The input argument 'new_model_dir' must be a string. It is the path" \
  + " where the fine tuned stardist model will be saved."

  assert isinstance(images, xr.DataArray), "the training samples should be a xr.DataArray"
  assert isinstance(ground_truth, xr.DataArray), "the ground thruth should be a xr.DataArray"

  assert  images.ndim == 4, "the training samples array must have 4 dimensions"
  assert  ground_truth.ndim == 3, "the training samples array must have 3 dimensions"

  assert "".join(images.dims) == "byxc", "the training samples axes order should be 'byxc', not '" + "".join(images.dims) +  "' as provided."
  assert "".join(ground_truth.dims) == "byx", "the ground truth samples axes order should be 'byx', not '" + "".join(ground_truth.dims) +  "' as provided."

  axes_dict = {"batch size": 0, "width": 1, "height": 2}

  for ks, vs in axes_dict.items():
    assert images.shape[vs] == ground_truth.shape[vs], "The training samples " \
    + "and the ground truth need to have the same " + ks + " : " \
    + str(images.shape[vs]) + " vs " + str(ground_truth.shape[vs])


def finetune_stardist(model_path, images, ground_truth, new_model_dir, weights_file=None):
  """
  model_path: String, path to pretrained model or pretrained model from the stardsit available
  images: list of tensors or single tensor? If a list of tensors, it would need to be ensured taht they all have same dims,
          or reconstruct to have same dims. Check the number of channels and check if the channels of the images coincide
          Also for a path, check that it has the needed files fo a stardist model
  ground_truth: list of tensors or single tensor? It needs to have the same type and size than images

  new_model_dir: directory where the new model will be saved, save one imput and output sample

  epochs and batch_size might have a warning for CPu if selected too large
  """
  assertions(model_path, images, ground_truth, new_model_dir)

  model = StarDist2D(None, model_path)
  if weights_file is not None:
    model.load_weights("weights_last.h5")

  # finetune on new data
  history = model.train(images, ground_truth, validation_data=(images, ground_truth))

  Path(new_model_dir).mkdir(parents=True, exist_ok=True)
  #model.keras_model.save(os.path.join(new_model_dir, "stardist_weights.h5"))
  model.export_TF(os.path.join(os.path.dirname(new_model_dir), "TF_SavedModel.zip"))

  return history.history