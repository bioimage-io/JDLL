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

"""Jython script that downloads the wanted model from the Bioimage.io repository,
downloads the engine and executes it on the sample image.
The example model downloaded is:
 - B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet
and can be found at: https://bioimage.io/#/?tags=B.%20Sutilist%20bacteria%20segmentation%20-%20Widefield%20microscopy%20-%202D%20UNet&id=10.5281%2Fzenodo.7261974

To run this script with the default parameters:
	python example-run-model-in-fiji.py

"""
from io.bioimage.modelrunner.bioimageio import BioimageioRepo
import sys
import os
from io.bioimage.modelrunner.engine.installation import EngineInstall


models_path = os.path.join(os. getcwd(), "models")
engine_path = os.path.join(os. getcwd(), "engines")
bmzModelName = "B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet"

if not os.path.exists(models_path) or not os.path.isdir(models_path):
    os.makedirs(models_path)
    
print("Connecting to the Bioimage.io repository")
br = BioimageioRepo.connect()
print("Downloading the Bioimage.io model: " + bmzModelName)
modelDir = br.downloadByName(bmzModelName, models_path)

print("Model downloaded at: " + modelDir)

print("Download the engine required for the model")
if not os.path.exists(engine_path) or not os.path.isdir(engine_path):
    os.makedirs(engine_path)

print("Installing JDLL engine")
success = EngineInstall.installEngineWithArgsInDir("tensorflow", 
						"2.7.0", True, Flase, engine_path)
if (success):
	print("Engine correctly installed at: " + engine_path)
else:
	print("Error with the engine installation.")
	return




