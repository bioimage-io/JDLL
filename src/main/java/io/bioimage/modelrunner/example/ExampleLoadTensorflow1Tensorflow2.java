/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 * #L%
 */
package io.bioimage.modelrunner.example;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.model.Model;
import io.bioimage.modelrunner.tensor.Tensor;

import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

/**
 * This is an example where a Tensorflow 2.4.1 and Tensorflow 1.15.0 are loaded on the same run.
 * 
 * The models used for this example are:
*  <ul>
*  <li><a href="https://bioimage.io/#/?tags=Neuron%20Segmentation%20in%202D%20EM%20%28Membrane%29&id=10.5281%2Fzenodo.5817052">Tf1</a></li>
*  <li><a href="https://github.com/stardist/stardist-icy/raw/main/src/main/resources/models/2D/dsb2018_paper.zip">Tf2</a></li>
*  </ul>
 * 
 * It also requires the installation of a TF1 and a TF2 engine
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class ExampleLoadTensorflow1Tensorflow2 {
	
	private static final String CWD = System.getProperty("user.dir");
	private static final String ENGINES_DIR = new File(CWD, "engines").getAbsolutePath();
	private static final String MODELS_DIR = new File(CWD, "models").getAbsolutePath();
	
	/**
	 * Loads a TF2 model and runs it
	 * @throws LoadEngineException if there is any error loading an engine
	 * @throws Exception if there is any exception running the model
	 */
	public static void loadAndRunTf2() throws LoadEngineException, Exception {
		// Tag for the DL framework (engine) that wants to be used
		String engine = "tensorflow_saved_model_bundle";
		// Version of the engine
		String engineVersion = "2.4.1";
		// Directory where all the engines are stored
		String enginesDir = ENGINES_DIR;
		// Path to the model folder
		String modelFolder = new File(MODELS_DIR, "dsb2018_paper").getAbsolutePath();
		// Path to the model source. The model source locally is the path to the source file defined in the 
		// yaml inside the model folder
		String modelSource = new File(MODELS_DIR, "dsb2018_paper").getAbsolutePath();
		// Whether the engine is supported by CPu or not
		boolean cpu = true;
		// Whether the engine is supported by GPU or not
		boolean gpu = false;
		// Create the EngineInfo object. It is needed to load the wanted DL framework
		// among all the installed ones. The EngineInfo loads the corresponding engine by looking
		// at the enginesDir at searching for the folder that is named satisfying the characteristics specified.
		// REGARD THAT the engine folders need to follow a naming convention
		EngineInfo engineInfo = createEngineInfo(engine, engineVersion, enginesDir, cpu, gpu);
		// Load the corresponding model
		Model model = loadModel(modelFolder, modelSource, engineInfo);
		// Create an image that will be the backend of the Input Tensor
		final ImgFactory< FloatType > imgFactory = new CellImgFactory<>( new FloatType(), 5 );
		final Img< FloatType > img1 = imgFactory.create( 1, 512, 512, 1 );
		// Create the input tensor with the nameand axes given by the rdf.yaml file
		// and add it to the list of input tensors
		Tensor<FloatType> inpTensor = Tensor.build("input", "bcyx", img1);
		List<Tensor<?>> inputs = new ArrayList<Tensor<?>>();
		inputs.add(inpTensor);
		
		// Create the output tensors defined in the rdf.yaml file with their corresponding 
		// name and axes and add them to the output list of tensors.
		/// Regard that output tensors can be built empty without allocating memory
		// or allocating memory by creating the tensor with a sample empty image, or by
		// defining the dimensions and data type
		final Img< FloatType > img2 = imgFactory.create( 1, 512, 512, 33 );
		Tensor<FloatType> outTensor = Tensor.build("output", "bcyx", img2);
		List<Tensor<?>> outputs = new ArrayList<Tensor<?>>();
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
		System.out.print("Success running Tensorflow 2!!");
	}

	/**
	 * Loads a TF1 model and runs it
	 * @throws LoadEngineException if there is any error loading an engine
	 * @throws Exception if there is any exception running the model
	 */
	public static void loadAndRunTf1() throws LoadEngineException, Exception {
		// Tag for the DL framework (engine) that wants to be used
		String engine = "tensorflow_saved_model_bundle";
		// Version of the engine
		String engineVersion = "1.15.0";
		// Directory where all the engines are stored
		String enginesDir = ENGINES_DIR;
		// Path to the model folder
		String modelFolder = new File(MODELS_DIR, "Neuron Segmentation in 2D EM (Membrane)_02022023_175546").getAbsolutePath();
		// Path to the model source. The model source locally is the path to the source file defined in the 
		// yaml inside the model folder
		String modelSource = modelFolder;
		// Whether the engine is supported by CPu or not
		boolean cpu = true;
		// Whether the engine is supported by GPU or not
		boolean gpu = false;
		// Create the EngineInfo object. It is needed to load the wanted DL framework
		// among all the installed ones. The EngineInfo loads the corresponding engine by looking
		// at the enginesDir at searching for the folder that is named satisfying the characteristics specified.
		// REGARD THAT the engine folders need to follow a naming convention
		EngineInfo engineInfo = createEngineInfo(engine, engineVersion, enginesDir, cpu, gpu);
		// Load the corresponding model
		Model model = loadModel(modelFolder, modelSource, engineInfo);
		// Create an image that will be the backend of the Input Tensor
		final ImgFactory< FloatType > imgFactory = new CellImgFactory<>( new FloatType(), 5 );
		final Img< FloatType > img1 = imgFactory.create( 1, 512, 512, 1 );
		// Create the input tensor with the nameand axes given by the rdf.yaml file
		// and add it to the list of input tensors
		Tensor<FloatType> inpTensor = Tensor.build("input0", "bcyx", img1);
		List<Tensor<?>> inputs = new ArrayList<Tensor<?>>();
		inputs.add(inpTensor);
		
		// Create the output tensors defined in the rdf.yaml file with their corresponding 
		// name and axes and add them to the output list of tensors.
		/// Regard that output tensors can be built empty without allocating memory
		// or allocating memory by creating the tensor with a sample empty image, or by
		// defining the dimensions and data type
		final Img< FloatType > img2 = imgFactory.create( 1, 512, 512, 1 );
		Tensor<FloatType> outTensor = Tensor.build("output0", "bcyx", img2);
		List<Tensor<?>> outputs = new ArrayList<Tensor<?>>();
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
		System.out.print("Success running Tensorflow 1!!");
	}
	
	/**
	 * Run the test
	 * @param <T>
	 * 	type of the tensors
	 * @param args
	 * 	arguments of the main method
	 * @throws LoadEngineException if there is any exception loading the engine
	 * @throws Exception if there is any exception in the tests
	 */
	public static < T extends RealType< T > & NativeType< T > > void main(String[] args) throws LoadEngineException, Exception {
		loadAndRunTf1();
		loadAndRunTf2();
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
		return EngineInfo.defineDLEngine(engine, engineVersion, enginesDir, cpu, gpu);
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
	public static Model loadModel(String modelFolder, String modelSource, EngineInfo engineInfo) throws LoadEngineException, Exception {
		
		Model model = Model.createDeepLearningModel(modelFolder, modelSource, engineInfo);
		model.loadModel();
		return model;
	}
}
