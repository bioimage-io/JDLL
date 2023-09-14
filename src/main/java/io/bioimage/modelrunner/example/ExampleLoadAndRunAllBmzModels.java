/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.example;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.engine.installation.EngineInstall;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.model.Model;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.versionmanagement.AvailableEngines;
import io.bioimage.modelrunner.versionmanagement.DeepLearningVersion;
import io.bioimage.modelrunner.versionmanagement.InstalledEngines;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

/**
 * This class tries to run every Bioimage.io model available
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class ExampleLoadAndRunAllBmzModels {
	
	private static final String CWD = System.getProperty("user.dir");
	private static final String ENGINES_DIR = new File(CWD, "engines").getAbsolutePath();
	private static final String MODELS_DIR = new File(CWD, "models").getAbsolutePath();
	
	/**
	 * Method that installs one engine compatible with the OS and Java version
	 * per DL framework and major version, this is installing Tf1, Tf2, Pytorch 1,
	 * Pytorch 2 and Onnx 17
	 */
	private static void installAllValidEngines() {
		EngineInstall installer = EngineInstall.createInstaller(ENGINES_DIR);
		installer.basicEngineInstallation();
	}

	/**
	 * 
	 * @param args
	 * 	main args, in this case nothing is needed
	 * @throws LoadEngineException if the engine fails to load the model
	 * @throws Exception if there is any error downloading the engines or the models
	 * 	or running the model
	 */
	public static void main(String[] args) throws LoadEngineException, Exception {
		
		installAllValidEngines();
		
		BioimageioRepo br = BioimageioRepo.connect();
		
		Map<Path, ModelDescriptor> bmzModelList = br.listAllModels(false);
		
		for (Entry<Path, ModelDescriptor> modelEntry : bmzModelList.entrySet()) {
			if (modelEntry.getValue().getWeights() == null)
				continue;
			String modelFolder = br.downloadByName(modelEntry.getValue().getName(), MODELS_DIR);
			Model model = Model.createBioimageioModel(modelFolder, ENGINES_DIR);
			model.loadModel();
		}

		// Tag for the DL framework (engine) that wants to be used
		String framework = "torchscript";
		// Version of the engine
		String engineVersion = "1.13.1";
		// Directory where all the engines are stored
		String enginesDir = ENGINES_DIR;
		// Download an engine that is ompatible with the model of interest
		downloadCPUEngine(framework, engineVersion, enginesDir);
		
		// Name of the model of interest from the Bioimage.io model repository
		String bmzModelName = "EnhancerMitochondriaEM2D";
		// Download the model of interest using its name
		String modelFolder = downloadBMZModel(bmzModelName, MODELS_DIR);
		
		// Path to the model source. The model source locally is the path to the source file defined in the 
		// yaml inside the model folder
		String modelSource = new File(modelFolder, "weights-torchscript.pt").getAbsolutePath();
		// Whether the engine is supported by CPU or not
		boolean cpu = true;
		// Check that the engine of interest is installed
		List<DeepLearningVersion> installedList = 
				InstalledEngines.checkEngineWithArgsInstalledForOS(framework, engineVersion, 
						cpu, null, enginesDir);
		// Get the first engine that fulfills the requirements and get whether
		// it supports GPU or not
		boolean gpu = installedList.get(0).getGPU();
		// Create the EngineInfo object. It is needed to load the wanted DL framework
		// among all the installed ones. The EngineInfo loads the corresponding engine by looking
		// at the enginesDir at searching for the folder that is named satisfying the characteristics specified.
		// REGARD THAT the engine folders need to follow a naming convention
		EngineInfo engineInfo = createEngineInfo(framework, engineVersion, enginesDir, cpu, gpu);
		// Load the corresponding model
		Model model = loadModel(modelFolder, modelSource, engineInfo);
		// Create an image that will be the backend of the Input Tensor
	
	}
	
	public static void loadAndRunModel(String modelFolder, ModelDescriptor descriptor) throws Exception {
		Model model = Model.createBioimageioModel(modelFolder, ENGINES_DIR);
		model.loadModel();
		List<Tensor<?>> inputs = new ArrayList<Tensor<?>>();
		List<Tensor<?>> outputs = new ArrayList<Tensor<?>>();
		final ImgFactory< FloatType > imgFactory = new ArrayImgFactory<>( new FloatType() );
		
		for ( TensorSpec it : descriptor.getInputTensors()) {
			String axesStr = it.getAxesOrder();
			String name = it.getName();
			it.getShape().getPatchMinimumSize()
		}
		
		final Img< FloatType > img1 = imgFactory.create( 1, 1, 512, 512 );
		// Create the input tensor with the nameand axes given by the rdf.yaml file
		// and add it to the list of input tensors
		Tensor<FloatType> inpTensor = Tensor.build("input0", "bcyx", img1);
		inputs.add(inpTensor);
		
		// Create the output tensors defined in the rdf.yaml file with their corresponding 
		// name and axes and add them to the output list of tensors.
		/// Regard that output tensors can be built empty without allocating memory
		// or allocating memory by creating the tensor with a sample empty image, or by
		// defining the dimensions and data type
		/*Tensor<FloatType> outTensor = Tensor.buildEmptyTensorAndAllocateMemory("output0", 
				"bcyx", 
				new long[] {1, 2, 512, 512}, 
				new FloatType());*/
		final Img< FloatType > img2 = imgFactory.create( 1, 2, 512, 512 );
		Tensor<FloatType> outTensor = Tensor.build("output0", "bcyx", img2);
		outputs.add(outTensor);
		
		// Run the model on the input tensors. THe output tensors 
		// will be rewritten with the result of the execution
		System.out.println(Util.average(Util.asDoubleArray(outputs.get(0).getData())));
		model.runModel(inputs, outputs);
		System.out.println(Util.average(Util.asDoubleArray(outputs.get(0).getData())));
		// The result is stored in the list of tensors "outputs"
		model.closeModel();
		inputs.stream().forEach(t -> t.close());
		outputs.stream().forEach(t -> t.close());
		System.out.print("Success!!");
	}
}
