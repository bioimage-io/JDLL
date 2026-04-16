/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.model.special.yolo;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat;
import io.bioimage.modelrunner.bioimageio.tiling.TileCalculator;
import io.bioimage.modelrunner.download.MultiFileDownloader;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.python.BioimageIoModelPytorchProtected;
import io.bioimage.modelrunner.model.python.DLModelPytorch;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.Constants;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

/**
 * Implementation of an API to run Cellpose models out of the box with little configuration.
 * 
 *
 *@author Carlos Garcia
 */
public class Yolo extends DLModelPytorch {
		
							
	private static final Map<String, Long> PRETRAINED_YOLO_MODELS;
	static {
		PRETRAINED_YOLO_MODELS = new HashMap<String, Long>();
		PRETRAINED_YOLO_MODELS.put("YOLO26n", 5_544_453L);
		PRETRAINED_YOLO_MODELS.put("YOLO26m", 44_255_705L);
		PRETRAINED_YOLO_MODELS.put("YOLO26x", 118_667_365L);
	}
	
	protected static final String LOAD_MODEL_CODE_ABSTRACT = ""
			+ "if 'denoise' not in globals().keys():" + System.lineSeparator()
			+ "  from cellpose import denoise" + System.lineSeparator()
			+ "  globals()['denoise'] = denoise" + System.lineSeparator()
			+ "if 'np' not in globals().keys():" + System.lineSeparator()
			+ "  import numpy as np" + System.lineSeparator()
			+ "  globals()['np'] = np" + System.lineSeparator()
			+ "if 'os' not in globals().keys():" + System.lineSeparator()
			+ "  import os" + System.lineSeparator()
			+ "  globals()['os'] = os" + System.lineSeparator()
			+ "if 'shared_memory' not in globals().keys():" + System.lineSeparator()
			+ "  from multiprocessing import shared_memory" + System.lineSeparator()
			+ "  globals()['shared_memory'] = shared_memory" + System.lineSeparator()
			+ "gpu_available = False" + System.lineSeparator() // TODO GPU
			+ ((IS_ARM) 
					? "" 
					: "from torch.backends import mps" + System.lineSeparator()
					+ "if mps.is_built() and mps.is_available():" + System.lineSeparator()
					+ "  gpu_available = True" + System.lineSeparator())
			+ MODEL_VAR_NAME + " = denoise.CellposeDenoiseModel(gpu=gpu_available, pretrained_model=r'%s')" + System.lineSeparator()
			+ "globals()['" + MODEL_VAR_NAME + "'] = " + MODEL_VAR_NAME + System.lineSeparator();

	/**
	 * Creates a new Cellpose.
	 *
	 * @param modelFile the modelFile parameter.
	 * @param callable the callable parameter.
	 * @param weightsPath the weightsPath parameter.
	 * @param kwargs the kwargs parameter.
	 * @param descriptor the descriptor parameter.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if there is any error building the environment
	 */
	protected Yolo(String modelFile, String callable, String weightsPath, 
			Map<String, Object> kwargs) throws BuildException, IOException {
		super(weightsPath, weightsPath, weightsPath, weightsPath, kwargs);
    	createPythonService();
	}
	
	// TODO add 3D
	protected <R extends RealType<R> & NativeType<R>> 
	List<Tensor<R>> checkInputTensors(List<Tensor<R>> inputTensors) {
		if (inputTensors.size() > 1)
			throw new IllegalArgumentException("The input tensor list should contain just one tensor");
		if (!inputTensors.get(0).getAxesOrderString().equals("xy") && !inputTensors.get(0).getAxesOrderString().equals("xyc"))
			throw new IllegalArgumentException("The input axes should be 'xyc'");

		long[] dims = inputTensors.get(0).getData().dimensionsAsLongArray();
		if (dims.length == 2) {
			FinalInterval interval = new FinalInterval(new long[3], new long[] {dims[0], dims[1], 1});
			IntervalView<R> nData = Views.interval(inputTensors.get(0).getData(), interval);
			inputTensors.set(0, Tensor.build(inputTensors.get(0).getName(), "xyc", nData));
		} else if (dims.length == 3 && dims[2] != 3 && dims[2] != 1)
			throw new IllegalArgumentException("Only 1 and 3 channel images supported. The provided input has " + dims[2]);
		return inputTensors;
	}
	
	protected <T extends RealType<T> & NativeType<T>> 
	List<Tensor<T>> checkOutputTensors(List<Tensor<T>> outputTensors) {
		// TODO 
		return outputTensors;
	}
	
	/**
	 * Run a Bioimage.io model and execute the tiling strategy in one go.
	 * The model needs to have been previously loaded with {@link #loadModel()}.
	 * This method does not execute pre- or post-processing, they
	 * need to be executed independently before or after
	 * 
	 * @param <T>
	 * 	ImgLib2 data type of the output images
	 * @param <R>
	 * 	ImgLib2 data type of the input images
	 * @param inputTensors
	 * 	list of the input tensors that are going to be inputed to the model
	 * @return the resulting tensors 
	 * @throws RunModelException if the model has not been previously loaded
	 * @throws IllegalArgumentException if the model is not a Bioimage.io model or if lacks a Bioimage.io
	 *  rdf.yaml specs file in the model folder. 
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> run(List<Tensor<R>> inputTensors) throws RunModelException {
		return super.run(checkInputTensors(inputTensors));
	}

	/**
	 * Run a Bioimage.io model and execute the tiling strategy in one go.
	 * The model needs to have been previously loaded with {@link #loadModel()}.
	 * This method does not execute pre- or post-processing, they
	 * need to be executed independently before or after
	 * 
	 * @param <T>
	 * 	ImgLib2 data type of the output images
	 * @param <R>
	 * 	ImgLib2 data type of the input images
	 * @param inputTensors
	 * 	list of the input tensors that are going to be inputed to the model
	 * @param outputTensors
	 * 	list of output tensors that are expected to be returned by the model
	 * @throws RunModelException if the model has not been previously loaded
	 * @throws IllegalArgumentException if the model is not a Bioimage.io model or if lacks a Bioimage.io
	 *  rdf.yaml specs file in the model folder. 
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	void run(List<Tensor<T>> inputTensors, List<Tensor<R>> outputTensors) throws RunModelException {
		super.run(checkInputTensors(inputTensors), checkOutputTensors(outputTensors));
	}
	
	/**
	 * Builds model code.
	 *
	 * @return the resulting string.
	 * @throws IOException if an I/O error occurs.
	 */
	protected String buildModelCode() throws IOException {
		String code = String.format(LOAD_MODEL_CODE_ABSTRACT, 
				//"False", // TODO GPU 
				this.weightsPath);
		return code;
	}
	
	protected <T extends RealType<T> & NativeType<T>> 
	String createInputsCode(List<RandomAccessibleInterval<T>> inRais, List<String> names) {
		String code = "";
		code += "created_shms = []" + System.lineSeparator();
		code += "try:" + System.lineSeparator();
		for (int i = 0; i < inRais.size(); i ++) {
			SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(inRais.get(i), false, false);
			code += codeToConvertShmaToPython(shma, names.get(i));
			inShmaList.add(shma);
		}
		String nameList = "[";
		String channelList = "[";
		for (int i = 0; i < inRais.size(); i ++) {
			nameList += names.get(i) + ", ";
			channelList += createChannelsArgCode(inRais.get(i)) + ", ";
		}
		nameList += "]";
		channelList += "]";
		code += createDiamCode(nameList, channelList);
		code += "  print(diameter)" + System.lineSeparator();
		code += "  " + OUTPUT_LIST_KEY + " = " + MODEL_VAR_NAME + ".eval(" + nameList + ", channels=" + channelList + ", ";
		code += "diameter=diameter)" + System.lineSeparator();;
		String closeEverythingWin = closeSHMWin();
		code += "  " + closeEverythingWin + System.lineSeparator();
		code += "except Exception as e:" + System.lineSeparator();
		code += "  " + closeEverythingWin + System.lineSeparator();
		code += "  raise e" + System.lineSeparator();
		code += ""
				+ SHMS_KEY + " = []" + System.lineSeparator()
				+ SHM_NAMES_KEY + " = []" + System.lineSeparator()
				+ DTYPES_KEY + " = []" + System.lineSeparator()
				+ DIMS_KEY + " = []" + System.lineSeparator()
				+ "globals()['" + SHMS_KEY + "'] = " + SHMS_KEY + System.lineSeparator()
				+ "globals()['" + SHM_NAMES_KEY + "'] = " + SHM_NAMES_KEY + System.lineSeparator()
				+ "globals()['" + DTYPES_KEY + "'] = " + DTYPES_KEY + System.lineSeparator()
				+ "globals()['" + DIMS_KEY + "'] = " + DIMS_KEY + System.lineSeparator();
		code += "handle_output_list(" + OUTPUT_LIST_KEY + ")" + System.lineSeparator();
		code += taskOutputsCode();
		return code;
	}
	
	/**
	 * Initialize a Cellpose model with the path to the model weigths.
	 * @param weightsPath
	 * 	path to the weights of a pretrained cellpose model
	 * @return an instance of a Stardist2D model ready to be used
     * @throws IOException If there's an I/O error.
	 * @throws BuildException if there is any error building the environment
	 */
	public static Yolo init(String weightsPath) throws IOException, BuildException {
		File wFile = new File(weightsPath);
		if (wFile.isDirectory() && new File(wFile, Constants.RDF_FNAME).isFile())
			return init(ModelDescriptorFactory.readFromLocalFile(new File(wFile, Constants.RDF_FNAME).getAbsolutePath()));
		if (!wFile.isFile())
			throw new IllegalArgumentException("The path provided does not correspond to an existing file: " + weightsPath);		        
        Yolo cellpose = new Yolo(null, null, weightsPath, null);
		return cellpose;
	}
	
	/**
	 * Initialize one of the "official" pretrained Stardist 2D models.
	 * By default, the model will be installed in the "models" folder inside the application
	 * @param pretrainedModel
	 * 	the name of the pretrained model. 
	 * @param install
	 * 	whether to force the download or to try to look if the model has already been installed before
	 * @return an instance of a pretrained Stardist2D model ready to be used
	 * @throws IOException if there is any error downloading the model, in the case it is needed
	 * @throws InterruptedException if the download of the model is stopped
	 * @throws ExecutionException if there is an error downloading the model
	 * @throws BuildException if there is any error building the environment
	 */
	public static Yolo fromPretained(String pretrainedModel, boolean install) throws IOException, InterruptedException, ExecutionException, BuildException {
		return fromPretained(pretrainedModel, new File("models").getAbsolutePath(), install);
	}
	
	/**
	 * Initialize one of the "official" pretrained cellpose ("cyto2", "cyto3"...) models or
	 * those available in the bioimage.io
	 * @param pretrainedModel
	 * 	the name of the pretrained model.
	 * @param modelsDir
	 * 	the directory where the model wants to be installed
	 * @param install
	 * 	whether to force the installation or to try to look if the model has already been installed before
	 * @return an instance of a pretrained Stardist2D model ready to be used
	 * @throws IOException if there is any error downloading the model, in the case it is needed
	 * @throws InterruptedException if the download of the model is stopped
	 * @throws ExecutionException if there is an error downloading the model
	 * @throws BuildException if there is any error building the environment
	 */
	public static Yolo fromPretained(String pretrainedModel, String modelsDir, boolean install) throws IOException, 
																					InterruptedException, ExecutionException, BuildException {
		return null;
	}
	
	/**
	 * Finds whether a pretrained Cellpose model is installed in the wanted directory
	 * 
	 * @param modelName
	 * 	the name of the model, it can be either the name of one of the official Cellpose models (cyto, cyto2, cyto3...)
	 * 	or a path to the weigths
	 * @param modelsDir
	 * 	the directory where we want to know whether the model is installed or not
	 * @return the path to the model if if exists, null otherwise
	 */
	public static String findPretrainedModelInstalled(String modelName, String modelsDir) {
		return null;
	}
	
	
	/**
	 * Example code that shows how to run a model with cellpose
	 * @param <T>
	 * 	method param
	 * @param args
	 * 	method param
	 * @throws IOException exception
	 * @throws InterruptedException exception
	 * @throws ExecutionException exception
	 * @throws LoadModelException exception
	 * @throws RunModelException exception
	 * @throws BuildException if there is any error launching the python process
	 */
	public static <T extends RealType<T> & NativeType<T>>
	void main(String[] args) throws IOException, InterruptedException, ExecutionException, LoadModelException, RunModelException, BuildException {
		Yolo model = Yolo.fromPretained("cyto2", false);
		model.loadModel();
		ArrayImg<FloatType, FloatArray> rai = ArrayImgs.floats(new long[] {512, 512, 3});
		List<RandomAccessibleInterval<FloatType>> rais = new ArrayList<RandomAccessibleInterval<FloatType>>();
		rais.add(rai);
		long tt = System.currentTimeMillis();
		List<RandomAccessibleInterval<T>> res = model.inference(rais);
		System.out.println(System.currentTimeMillis() - tt);
		tt = System.currentTimeMillis();
		List<RandomAccessibleInterval<T>> rees = model.inference(rais);
		System.out.println(System.currentTimeMillis() - tt);
		model.close();
		System.out.println(false);
	}
}
