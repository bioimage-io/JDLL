# Copyright (C) 2023 Institut Pasteur.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Jython script that downloads the wanted Pytorch model from the Bioimage.io repository,
downloads the engine and executes it on the sample image.
The example model downloaded is:
 - B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet
and can be found at: https://bioimage.io/#/?type=all&tags=Mitochondria%20resolution%20enhancement%20Wasserstein%20GAN&id=10.5281%2Fzenodo.7786492

To run this script with the default parameters:
	python example-run-pytorch-model-in-fiji.py


"""
from io.bioimage.modelrunner.engine.installation import EngineInstall
from io.bioimage.modelrunner.bioimageio import BioimageioRepo
from io.bioimage.modelrunner.model import Model
from io.bioimage.modelrunner.tensor import Tensor
from io.bioimage.modelrunner.versionmanagement import AvailableEngines

import sys
import os

from ij import IJ

from net.imglib2.img.display.imagej import ImageJFunctions
from net.imglib2.view import Views


models_path = os.path.join(os. getcwd(), "models")
engine_path = os.path.join(os. getcwd(), "engines")
bmzModelName = "Mitochondria resolution enhancement Wasserstein GAN"

if not os.path.exists(models_path) or not os.path.isdir(models_path):
    os.makedirs(models_path)
    
print("Connecting to the Bioimage.io repository")
br = BioimageioRepo.connect()
print("Downloading the Bioimage.io model: " + bmzModelName)
model_fn = br.downloadByName(bmzModelName, models_path)

print("Model downloaded at: " + model_fn)

print("Download the engine required for the model")
if not os.path.exists(engine_path) or not os.path.isdir(engine_path):
    os.makedirs(engine_path)

print("Installing JDLL engine")
supportedList = AvailableEngines.getEnginesForOsByParams("pytorch", "1.13.1", True, None)
gpu = supportedList[0].getGPU()
success = EngineInstall.installEngineWithArgsInDir("pytorch", 
						"1.13.1", True, gpu, engine_path)
if (success):
	print("Engine correctly installed at: " + engine_path)
else:
	raise Error("Error with the engine installation.")

imp = IJ.openImage(os.path.join(model_fn, "sample_input_0.tif"))
imp.show()

wrapImg = ImageJFunctions.convertFloat(imp)
wrapImg = Views.addDimension(wrapImg, 0, 0)
wrapImg = Views.addDimension(wrapImg, 0, 0)
wrapImg = Views.permute(wrapImg, 0, 2)
wrapImg = Views.permute(wrapImg, 1, 3)

inputTensor = Tensor.build("input", "bcxy", wrapImg)
outputTensor = Tensor.buildEmptyTensor("output", "bcxy")


model = Model.createBioimageioModel(model_fn, engine_path)
print("Loading model")
model.loadModel()
print("Running model")

model.runModel([inputTensor], [outputTensor])
ImageJFunctions.show( Views.dropSingletonDimensions(outputTensor.getData()) )
print("Display output")
model.closeModel()

inputTensor.close()
outputTensor.close()
