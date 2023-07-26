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

import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.model.Model;
import io.bioimage.modelrunner.tensor.Tensor;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

/**
 * This is an example of the library that runs a Deep Learning model on a
 * supported engine locally on your computer. Regard that in order to this
 * example to work, a Deep Learning model needs to be downloaded from the
 * Bioimage.io repo and a Java Deep Learning framework needs to be installed
 * too, in your user directory. This example uses the torchscript/DeepImageJ
 * model <a href=
 * "https://bioimage.io/#/?tags=10.5281%2Fzenodo.6406756&id=10.5281%2Fzenodo.6406756">here</a>.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class ExampleLoadAndRunModel {
	
	private static final String CWD = System.getProperty("user.dir");
	private static final String ENGINES_DIR = new File(CWD, "engines").getAbsolutePath();
	private static final String MODELS_DIR = new File(CWD, "models").getAbsolutePath();

	/**
	 * 
	 * @param <T>
	 * @param args
	 * @throws LoadEngineException
	 * @throws Exception
	 */
	public static < T extends RealType< T > & NativeType< T > > void main(String[] args) throws LoadEngineException, Exception {
		// Tag for the DL framework (engine) that wants to be used
		String engine = "torchscript";
		// Version of the engine
		String engineVersion = "1.13.1";
		// Directory where all the engines are stored
		String enginesDir = ENGINES_DIR;
		// Path to the model folder
		String modelFolder = new File(MODELS_DIR, "EnhancerMitochondriaEM2D_13012023_130426").getAbsolutePath();
		// Path to the model source. The model source locally is the path to the source file defined in the 
		// yaml inside the model folder
		String modelSource = new File(modelFolder, "weights-torchscript.pt").getAbsolutePath();
		// Whether the engine is supported by CPu or not
		boolean cpu = true;
		// Whether the engine is supported by GPU or not
		boolean gpu = true;
		// Create the EngineInfo object. It is needed to load the wanted DL framework
		// among all the installed ones. The EngineInfo loads the corresponding engine by looking
		// at the enginesDir at searching for the folder that is named satisfying the characteristics specified.
		// REGARD THAT the engine folders need to follow a naming convention
		EngineInfo engineInfo = createEngineInfo(engine, engineVersion, enginesDir, cpu, gpu);
		// Load the corresponding model
		Model model = loadModel(modelFolder, modelSource, engineInfo);
		// Create an image that will be the backend of the Input Tensor
		final ImgFactory< FloatType > imgFactory = new ArrayImgFactory<>( new FloatType() );
		final Img< FloatType > img1 = imgFactory.create( 1, 1, 512, 512 );
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
		/*Tensor<FloatType> outTensor = Tensor.buildEmptyTensorAndAllocateMemory("output0", 
				"bcyx", 
				new long[] {1, 2, 512, 512}, 
				new FloatType());*/
		final Img< FloatType > img2 = imgFactory.create( 1, 2, 512, 512 );
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
		System.out.print("Success!!");
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
		return EngineInfo.defineDLEngine(engine, engineVersion, cpu, gpu, enginesDir);
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
	 * @throws Exception if anything fails loading the model
	 */
	public static Model loadModel(String modelFolder, String modelSource, EngineInfo engineInfo) throws LoadEngineException, Exception {
		
		Model model = Model.createDeepLearningModel(modelFolder, modelSource, engineInfo);
		model.loadModel();
		return model;
	}
}
