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

"""Jython script that downloads the wanted DL engine into the wanted folder.
The default example engine downloaded is:
 - Tensorflow 2.7.0 for CPU and GPU


To download and install the default example engine, run the script with no parameters:
	
	python example-download-engine.py

In order to download the wanted engine in the wanted directory,
please provide *five* parameters:. The first one should be the framework
name, the second one the frmaework version, the third one True or False whether
CPU is supported or not, the fourth one True or False whether GPU is supported
or not, and the fifth should be the directory where the engine is installed:
	
	python example-download-engine.py framework_name version true_or_false true_or_false


A list of the supported DL frameworks and versions can be found at:
https://github.com/bioimage-io/JDLL/wiki/List-of-supported-engines

Executing the script without parameters is equal to executing it with the
following parameters:

	python example-download-engine.py tensorflow 2.7.0 True True models
"""

from io.bioimage.modelrunner.bioimageio import BioimageioRepo
import sys
import os

framework_list = ["tensorflow", "tensorflow_saved_model_bundle",
					"torchscript", "pytorch", "onnx"]
full_path = os.path.join(os. getcwd(), "engines")
framework = "tensorflow"
version = 2.7.0
cpu = True
gpu = True

if len(sys.argv) != 1 and len(sys.argv) == 6:
	raise TypeError("Script only works either when no arguments are provided or "\
					+ "when 6 arguments are provided.")

if len(sys.argv) == 6 and type(sys.argv[3]) == bool:
	expected_types = [str, str, bool, bool, str]
	for i, arg in enumerate(sys.argv, start=1):
        if not isinstance(arg, expected_types[i]):
            raise TypeError(f"Argument {i} is not of the correct data type ({expected_types[i]}).")
elif len(sys.argv) == 6 and sys.argv[1] not in framework_list:
	raise TypeError("First argument for the script should be among the supported" \
	 					+ " DL frameworks: " + str(framework_list))
elif len(sys.argv) == 6:
	framework = sys.argv[1]
	version = sys.argv[2]
	cpu = sys.argv[3]
	gpu = sys.argv[4]
	full_path = sys.argv[5]


print("Connecting to the Bioimage.io repository")
br = BioimageioRepo.connect()
print("Downloading the Bioimage.io model: " + bmzModelName)
modelDir = br.downloadByName(bmzModelName, full_path)

print("Model downloaded at: " + modelDir)
