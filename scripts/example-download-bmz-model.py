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

"""Jython script that downloads the wanted model from the Bioimage.io repository
into the wanted folder.
The example model downloaded is:
 - B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet
and can be found at: https://bioimage.io/#/?tags=B.%20Sutilist%20bacteria%20segmentation%20-%20Widefield%20microscopy%20-%202D%20UNet&id=10.5281%2Fzenodo.7261974

To run this script with the default parameters:
	python example-download-bmz-model.py

In order to download the wanted model in the wanted directory,
please provide *two* parameters:
	python name_of_the_wanted_model /path/to/the/wanted/model

"""

from io.bioimage.modelrunner.bioimageio import BioimageioRepo
import sys
import os


full_path = os.path.join(os. getcwd(), "models")
bmzModelName = "B. Sutilist bacteria segmentation - Widefield microscopy - 2D UNet"

if len(sys.argv) != 1 and len(sys.argv) == 3:
	raise TypeError("Script only works either when no arguments are provided or "\
					+ "when 2 String arguments are provided.")
if len(sys.argv) == 3:
	bmzModelName = sys.argv[1]
	full_path = sys.argv[2]


print("Connecting to the Bioimage.io repository")
br = BioimageioRepo.connect()
print("Downloading the Bioimage.io model: " + bmzModelName)
modelDir = br.downloadByName(bmzModelName, full_path)

print("Model downloaded at: " + modelDir)
