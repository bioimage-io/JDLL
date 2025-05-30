/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.engine.installation.EngineInstall;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.model.java.DLModelJava;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.versionmanagement.AvailableEngines;
import io.bioimage.modelrunner.versionmanagement.DeepLearningVersion;
import io.bioimage.modelrunner.versionmanagement.InstalledEngines;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

/**
 * This is an example where a Pytorch 1.13.1 and Pytorch 2.0.0 are loaded on the same run.
 * 
 * The models used for this example is:
 * <ul>
 * <li><a href="https://bioimage.io/#/?tags=10.5281%2Fzenodo.6406756&id=10.5281%2Fzenodo.6406756t">EnhancerMitochondriaEM2D (both Pt1 and 2)</a></li>
 * </ul>
 * 
 * It also requires the installation of a Pytorch 1 and a Pytorch 2 engine.
 * 
 * The example code downloads all the needed artifacts, thus executing the whole
 * example might take some time.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class ExampleLoadPytorch1Pytorch2 {
	
	private static final String CWD = System.getProperty("user.dir");
	private static final String ENGINES_DIR = new File(CWD, "engines").getAbsolutePath();
	private static final String MODELS_DIR = new File(CWD, "models").getAbsolutePath();
	
	/**
	 * Run the test
	 * @param args
	 * 	arguments of the main method
	 * @throws LoadEngineException if there is any exception loading the engine
	 * @throws Exception if there is any exception in the tests
	 */
	public static void main(String[] args) throws LoadEngineException, Exception {
		
		// Name of the model of interest from the Bioimage.io model repository
		String bmzModelName = "EnhancerMitochondriaEM2D";
		// Download the model of interest using its name
		String modelFolder = downloadBMZModel(bmzModelName, MODELS_DIR);
		
		// Path to the model source. The model source locally is the path to the source file defined in the 
		// yaml inside the model folder
		String modelSource = new File(modelFolder, "weights-torchscript.pt").getAbsolutePath();
		loadAndRunPt1(modelFolder, modelSource);
		loadAndRunPt2(modelFolder, modelSource);
		System.out.println("Great success!");
	}
	
	/**
	 * Loads a Pytorch 2 model and runs it
	 * @param <T>
	 * 	ImgLib2 input data type
	 * @param <R>
	 * 	ImgLib2 output data type, can be the same as the input
	 * @param modelFolder
	 * 	folder of the model
	 * @param modelSource
	 * 	source of the model
	 * @throws LoadEngineException if there is any error loading an engine
	 * @throws Exception if there is any exception running the model
	 */
	public static <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	void loadAndRunPt2(String modelFolder, String modelSource) throws LoadEngineException, Exception {
		// Tag for the DL framework (engine) that wants to be used
		String framework = "torchscript";
		// Version of the engine
		String engineVersion = "2.0.0";
		// Directory where all the engines are stored
		String enginesDir = ENGINES_DIR;
		// Download an engine that is ompatible with the model of interest
		downloadCPUEngine(framework, engineVersion, enginesDir);
		
		// Whether the engine is supported by CPu or not
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
		// Load the corresponding model, for Pytorch model_source arg is not needed
		try (DLModelJava model = loadModel(modelFolder, modelSource, engineInfo);) {
			// Create an image that will be the backend of the Input Tensor
			final ImgFactory< FloatType > imgFactory = new ArrayImgFactory<>( new FloatType() );
			final Img< FloatType > img1 = imgFactory.create( 1, 1, 512, 512 );
			// Create the input tensor with the nameand axes given by the rdf.yaml file
			// and add it to the list of input tensors
			Tensor<FloatType> inpTensor = Tensor.build("input0", "bcyx", img1);
			List<Tensor<FloatType>> inputs = new ArrayList<Tensor<FloatType>>();
			inputs.add(inpTensor);
			
			// Create the output tensors defined in the rdf.yaml file with their corresponding 
			// name and axes and add them to the output list of tensors.
			/// Regard that output tensors can be built empty without allocating memory
			// or allocating memory by creating the tensor with a sample empty image, or by
			// defining the dimensions and data type
			/*Tensor<FloatType> outTensor = Tensor.buildBlankTensor("output0", 
					"bcyx", 
					new long[] {1, 2, 512, 512}, 
					new FloatType());*/
			final Img< FloatType > img2 = imgFactory.create( 1, 2, 512, 512 );
			Tensor<FloatType> outTensor = Tensor.build("output0", "bcyx", img2);
			List<Tensor<FloatType>> outputs = new ArrayList<Tensor<FloatType>>();
			outputs.add(outTensor);
			
			// Run the model on the input tensors. THe output tensors 
			// will be rewritten with the result of the execution
			System.out.println(Util.average(Util.asDoubleArray(outputs.get(0).getData())));
			model.run(inputs, outputs);
			System.out.println(Util.average(Util.asDoubleArray(outputs.get(0).getData())));
			// The result is stored in the list of tensors "outputs"
			inputs.stream().forEach(t -> t.close());
			outputs.stream().forEach(t -> t.close());
			System.out.println("Success running Pytorch 2!!");
		}
	}

	/**
	 * Loads a Pytorch 1 model and runs it
	 * @param <T>
	 * 	ImgLib2 input data type
	 * @param <R>
	 * 	ImgLib2 output data type, can be the same as the input
	 * @param modelFolder
	 * 	folder of the model
	 * @param modelSource
	 * 	source of the model
	 * @throws LoadEngineException if there is any error loading an engine
	 * @throws Exception if there is any exception running the model
	 */
	public static <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	void loadAndRunPt1(String modelFolder, String modelSource) throws LoadEngineException, Exception {
		// Tag for the DL framework (engine) that wants to be used
		String framework = "torchscript";
		// Version of the engine
		String engineVersion = "1.13.1";
		// Directory where all the engines are stored
		String enginesDir = ENGINES_DIR;
		// Download an engine that is ompatible with the model of interest
		downloadCPUEngine(framework, engineVersion, enginesDir);
		
		// Whether the engine is supported by CPu or not
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
		// Load the corresponding model, for Pytorch the arg model_source is not needed
		try (DLModelJava model = loadModel(modelFolder, modelSource, engineInfo);) {
			// Create an image that will be the backend of the Input Tensor
			final ImgFactory< FloatType > imgFactory = new ArrayImgFactory<>( new FloatType() );
			final Img< FloatType > img1 = imgFactory.create( 1, 1, 512, 512 );
			// Create the input tensor with the nameand axes given by the rdf.yaml file
			// and add it to the list of input tensors
			Tensor<FloatType> inpTensor = Tensor.build("input0", "bcyx", img1);
			List<Tensor<FloatType>> inputs = new ArrayList<Tensor<FloatType>>();
			inputs.add(inpTensor);
			
			// Create the output tensors defined in the rdf.yaml file with their corresponding 
			// name and axes and add them to the output list of tensors.
			/// Regard that output tensors can be built empty without allocating memory
			// or allocating memory by creating the tensor with a sample empty image, or by
			// defining the dimensions and data type
			/*Tensor<FloatType> outTensor = Tensor.buildBlankTensor("output0", 
					"bcyx", 
					new long[] {1, 2, 512, 512}, 
					new FloatType());*/
			final Img< FloatType > img2 = imgFactory.create( 1, 2, 512, 512 );
			Tensor<FloatType> outTensor = Tensor.build("output0", "bcyx", img2);
			List<Tensor<FloatType>> outputs = new ArrayList<Tensor<FloatType>>();
			outputs.add(outTensor);
			
			// Run the model on the input tensors. THe output tensors 
			// will be rewritten with the result of the execution
			System.out.println(Util.average(Util.asDoubleArray(outputs.get(0).getData())));
			model.run(inputs, outputs);
			System.out.println(Util.average(Util.asDoubleArray(outputs.get(0).getData())));
			// The result is stored in the list of tensors "outputs"
			inputs.stream().forEach(t -> t.close());
			outputs.stream().forEach(t -> t.close());
			System.out.println("Success running Pytorch 1!!");
		}
	}
	
	/**
	 * Method that creates the {@link EngineInfo} object.
	 * @param engine
	 * 	tag of the Deep Learning framework as definde in the bioimage.io
	 * @param engineVersion
	 * 	version of the Deep LEarning framework
	 * @param enginesDir
	 * 	directory where all the Deep Learning frameworks are installed
	 * @param cpu
	 * 	whether the engine is supported by CPU or not
	 * @param gpu
	 * 	whether the engine is supported by GPU or not
	 * @return an {@link EngineInfo} object to load a DL model
	 */
	public static EngineInfo createEngineInfo(String engine, String engineVersion, 
			String enginesDir, boolean cpu, boolean gpu) {
		return EngineInfo.defineCompatibleDLEngine(engine, engineVersion, cpu, gpu, enginesDir);
	}
	
	/**
	 * Load the wanted model
	 * @param modelFolder
	 * 	path to the model folder downloaded
	 * @param modelSource
	 * 	local path to the source file of the model
	 * @param engineInfo
	 * 	Object containing the needed info about the Deep Learning 
	 * 	framework compatible with the wanted model
	 * @return a loaded DL model
	 * @throws LoadEngineException if there is any error loading the model
	 * @throws Exception if there is any exception loading the model
	 */
	public static DLModelJava loadModel(String modelFolder, String modelSource, EngineInfo engineInfo) throws LoadEngineException, Exception {
		
		DLModelJava model = DLModelJava.createModel(modelFolder, modelSource, engineInfo);
		model.loadModel();
		return model;
	}
	
	/**
	 * Downloads the engine defined by the framework and engineVersion
	 * arguments that is supported on the CPU
	 * @param framework
	 * 	DL framework of interest
	 * @param engineVersion
	 * 	version of the DL framework of interest
	 * @param enginesDir
	 * 	directory where the engine is going to be installed
	 * @throws IOException if the engine is not installed correctly or no
	 * 	engine with the criteria is found
	 * @throws InterruptedException if the engine download is interrupted
	 * @throws ExecutionException if there si any error during the installation
	 */
	public static void downloadCPUEngine(String framework, String engineVersion,
			String enginesDir) throws IOException, InterruptedException, ExecutionException {
		// Check if there is any engine supported by JDLL that fulfils the
		// framework and version requirements that also runs on CPU
		List<DeepLearningVersion> possibleEngines = 
				AvailableEngines.getEnginesForOsByParams(framework, engineVersion, true, null);
		// Try to install the first match that fits the requirements, any other 
		// match could have been used too.
		EngineInstall.installEngineInDir(possibleEngines.get(0), enginesDir);
	}
	
	/**
	 * Download a model from the Bioimage.io repository selecting it by its full
	 * name. The model is downloaded into the wanted directory.
	 * 
	 * @param bmzModelName
	 * 	name of the model of interest
	 * @param modelsDir
	 * 	directory where the model is downloaded
	 * @return the path to the model downloaded. The path its model folder.
	 * @throws IOException if there is any error downloading the model or
	 * 	it does not exist
	 * @throws InterruptedException if the download is interrupted
	 */
	public static String downloadBMZModel(String bmzModelName, String modelsDir) throws IOException, InterruptedException {
		// Create an instance of the BioimageRepo object
		BioimageioRepo br = BioimageioRepo.connect();
		return br.downloadByName(bmzModelName, modelsDir);
	}
}
