import xarray as xr
from stardist.models import StarDist2D

import os
import shutil

def assertions(model, images, ground_truth, new_model_dir):

  assert isinstance(model, str), "The input argument 'model' must be a string, either the name" \
  + " of one of the default pre-trained Stardist models or the directory to a pre-trained Stardist model"
  
  assert isinstance(new_model_dir, str), "The input argument 'new_model_dir' must be a string. It is the path" \
  + " where the fine tuned stardist model will be saved."

  assert isinstance(images, xr.DataArray), "the training samples should be a xr.DataArray"
  assert isinstance(ground_truth, xr.DataArray), "the ground thruth should be a xr.DataArray"

  assert  images.ndim == 4, "the training samples array must have 4 dimensions"
  assert  ground_truth.ndim == 4, "the training samples array must have 4 dimensions"

  assert "".join(images.dims) == "bcyx", "the training samples axes order should be 'bcyx', not '" + "".join(images.dims) +  "' as provided."
  assert "".join(ground_truth.dims) == "bcyx", "the ground truth samples axes order should be 'bcyx', not '" + "".join(ground_truth.dims) +  "' as provided."

  axes_dict = {"batch size": 0, "width": 3, "height": 2}

  for ks, vs in axes_dict.items():
    assert images.shape[vs] == ground_truth.shape[vs], "The training samples " \
    + "and the ground truth need to have the same " + ks + " : " 
    + images.shape[vs] + " vs " + ground_truth.shape[vs]


def finetune_stardist(model, images, ground_truth, new_model_dir, epochs=5, lr=1e-5, batch_size=16):
  """
  model: String, path to pretrained model or pretrained model from the stardsit available
  images: list of tensors or single tensor? If a list of tensors, it would need to be ensured taht they all have same dims,
          or reconstruct to have same dims. Check the number of channels and check if the channels of the images coincide
          Also for a path, check that it has the needed files fo a stardist model
  ground_truth: list of tensors or single tensor? It needs to have the same type and size than images

  new_model_dir: directory where the new model will be saved, save one imput and output sample

  epochs and batch_size might have a warning for CPu if selected too large
  """
  assertions(model, images, ground_truth, new_model_dir)
  if ()

  model = StarDist2D(None, model)
  # change some training params 
  model.config.train_patch_size = (images.shape["width"], images.shape["height"])
  model.config.train_batch_size = images.shape["batch size"] 
  model.config.train_learning_rate = lr
  model.config.train_epochs = epochs

  # finetune on new data
  history = model.train(X,Y, validation_data=(X,Y))

  return history.loss, history.rest_losses