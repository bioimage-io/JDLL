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
/**
 * 
 */
package io.bioimage.modelrunner.model;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.tiling.ImageInfo;
import io.bioimage.modelrunner.bioimageio.tiling.TileCalculator;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.bioimageio.description.exceptions.ModelSpecsException;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat;
import io.bioimage.modelrunner.bioimageio.tiling.TileInfo;
import io.bioimage.modelrunner.bioimageio.tiling.TileMaker;
import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.processing.Processing;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.versionmanagement.InstalledEngines;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;

/**
 * Class that manages a Deep Learning model to load it and run it.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class BioimageIoModel extends DLModel
{
	/**
	 * Whether the model is created for the bioengine or not
	 */
	private boolean bioengine = false;
	/**
	 * Object containing the information of the rdf.yaml file of a Bioimage.io model
	 */
	protected ModelDescriptor descriptor;
	/**
	 * Calculates the tile sizes depending on the model specs
	 */
	protected TileCalculator tileCalculator;

	/**
	 * Construct the object model with all the needed information to load a
	 * model and make inference
	 * 
	 * @param engineInfo
	 *            informaton needed about the model
	 * @param modelFolder
	 *            directory where of the model folder
	 * @param modelSource
	 *            name of the actual model file (.pt for torchscript)
	 * @param classLoader
	 *            parent ClassLoader of the engine (can be null)
	 * @throws LoadEngineException
	 *             if there is an error finding the Deep LEarningn interface
	 *             that connects with the DL libraries
	 * @throws MalformedURLException if the JAR files are not well defined in the .json file
	 * @throws IOException if there is any error finding the engines in the system
	 * @throws IllegalStateException if any of the engines has been incorrectly modified
	 */
	protected BioimageIoModel( EngineInfo engineInfo, String modelFolder, String modelSource, ClassLoader classLoader )
			throws LoadEngineException, MalformedURLException, IllegalStateException, IOException
	{
		super(engineInfo, modelFolder, modelSource, classLoader);
	}
	
	public void setTiling(boolean doTiling) {
		this.tiling = false;
	}
	
	/**
	 * Load a model from the bioimage.io directly. Just providing the path to the
	 * folder where the rdf.yaml is, no extra info is needed as it is read from the
	 * rdf.yaml file
	 * To successfully create a Bioiamge.io model, it is required that there is installed
	 * at least one of the engines needed to load at least one of the weight formats
	 * supported by the model. Only the major version needs to be the same (Tensorflow 1 != Tensorflow 2).
	 * 
	 * @param bmzModelFolder
	 * 	folder where the bioimage.io model is located (parent folder of the rdf.yaml file)
	 * @param classloader
	 * 	Parent ClassLoader of the engine (can be null). Almost the same method as 
	 *  Model.createBioimageioModel( String bmzModelFolder, String enginesFolder ). 
	 *  The only difference is that this method can choose the parent ClassLoader for the engine. 
	 *  JDLL creates a separate ChildFirst-ParentLast CustomClassLoader for each of the 
	 *  engines loaded to avoid conflicts between them. In order to have access to the 
	 *  classes of the main ClassLoader the ChildFirst-ParentLast CustomClassLoader needs a parent. 
	 *  If no classloader argument is provided the parent ClassLoader will be the Thread's 
	 *  context ClassLoader (Thread.currentThread().getContextClassLoader()).
	 *  
	 *  The classloader argument is usually not needed, but for some softwares 
	 *  such as Icy, that have a custom management of ClassLoaders it is necessary.
	 * @return a model ready to be loaded
	 * @throws LoadEngineException if there is any error loading the DL framework
	 * @throws IOException if there is any error finding the engines in the system
	 * @throws ModelSpecsException if the rdf.yaml file has some at least a field which does not comply with the Bioiamge.io constraints
	 */
	public static BioimageIoModel createBioimageioModel(String bmzModelFolder, ClassLoader classloader)
			throws LoadEngineException, ModelSpecsException, IOException {
		return createBioimageioModel(bmzModelFolder, InstalledEngines.getEnginesDir(), classloader);
	}
	
	/**
	 * Load a model from the bioimage.io directly. Just providing the path to the
	 * folder where the rdf.yaml is, no extra info is needed as it is read from the
	 * rdf.yaml file
	 * To successfully create a Bioiamge.io model, it is required that there is installed
	 * at least one of the engines needed to load at least one of the weight formats
	 * supported by the model. Only the major version needs to be the same (Tensorflow 1 != Tensorflow 2).
	 * 
	 * @param bmzModelFolder
	 * 	folder where the bioimage.io model is located (parent folder of the rdf.yaml file)
	 * @return a model ready to be loaded
	 * @throws LoadEngineException if there is any error loading the DL framework
	 * @throws IOException if there is any error finding the engines in the system
	 * @throws ModelSpecsException if the rdf.yaml file has some at least a field which does not comply with the Bioiamge.io constraints
	 */
	public static BioimageIoModel createBioimageioModel(String bmzModelFolder)
			throws ModelSpecsException, LoadEngineException, IOException {
		return createBioimageioModel(bmzModelFolder, InstalledEngines.getEnginesDir());
	}
	
	/**
	 * Load a model from the bioimage.io directly. Just providing the path to the
	 * folder where the rdf.yaml is, no extra info is needed as it is read from the
	 * rdf.yaml file.
	 * To successfully create a Bioiamge.io model, it is required that there is installed
	 * at least one of the engines needed to load at least one of the weight formats
	 * supported by the model. Only the major version needs to be the same (Tensorflow 1 != Tensorflow 2).
	 * 
	 * @param bmzModelFolder
	 * 	folder where the bioimage.io model is located (parent folder of the rdf.yaml file)
	 * @param enginesFolder
	 * 	directory where all the engine (DL framework) folders are downloaded
	 * @return a model ready to be loaded
	 * @throws LoadEngineException if there is any error loading the DL framework
	 * @throws IOException if there is any error finding the engines in the system
	 * @throws ModelSpecsException if the rdf.yaml file has some at least a field which does not comply with the Bioiamge.io constraints
	 */
	public static BioimageIoModel createBioimageioModel(String bmzModelFolder, String enginesFolder) 
			throws ModelSpecsException, LoadEngineException, IOException {
		return createBioimageioModel(bmzModelFolder, enginesFolder, null);
	}
	
	/**
	 * Load a model from the bioimage.io directly. Just providing the path to the
	 * folder where the rdf.yaml is, no extra info is needed as it is read from the
	 * rdf.yaml file.
	 * To successfully create a Bioiamge.io model, it is required that there is installed
	 * at least one of the engines needed to load at least one of the weight formats
	 * supported by the model. Only the major version needs to be the same (Tensorflow 1 != Tensorflow 2).
	 * 
	 * @param bmzModelFolder
	 * 	folder where the bioimage.io model is located (parent folder of the rdf.yaml file)
	 * @param enginesFolder
	 * 	directory where all the engine (DL framework) folders are downloaded
	 * @param classloader
	 * 	Parent ClassLoader of the engine (can be null). Almost the same method as 
	 *  Model.createBioimageioModel( String bmzModelFolder, String enginesFolder ). 
	 *  The only difference is that this method can choose the parent ClassLoader for the engine. 
	 *  JDLL creates a separate ChildFirst-ParentLast CustomClassLoader for each of the 
	 *  engines loaded to avoid conflicts between them. In order to have access to the 
	 *  classes of the main ClassLoader the ChildFirst-ParentLast CustomClassLoader needs a parent. 
	 *  If no classloader argument is provided the parent ClassLoader will be the Thread's 
	 *  context ClassLoader (Thread.currentThread().getContextClassLoader()).
	 *  
	 *  The classloader argument is usually not needed, but for some softwares 
	 *  such as Icy, that have a custom management of ClassLoaders it is necessary.
	 * @return a model ready to be loaded
	 * @throws LoadEngineException if there is any error loading the DL framework
	 * @throws IOException if there is any error finding the engines in the system
	 * @throws ModelSpecsException if the rdf.yaml file has some at least a field which does not comply with the Bioiamge.io constraints
	 */
	public static BioimageIoModel createBioimageioModel(String bmzModelFolder, String enginesFolder, ClassLoader classloader) 
			throws LoadEngineException, IOException, ModelSpecsException {
		Objects.requireNonNull(bmzModelFolder);
		Objects.requireNonNull(enginesFolder);
		if (new File(bmzModelFolder, Constants.RDF_FNAME).isFile() == false)
			throw new IOException("A Bioimage.io model folder should contain its corresponding rdf.yaml file.");
		ModelDescriptor descriptor = 
			ModelDescriptorFactory.readFromLocalFile(bmzModelFolder + File.separator + Constants.RDF_FNAME);
		String modelSource = null;
		List<WeightFormat> modelWeights = descriptor.getWeights().gettAllSupportedWeightObjects();
		EngineInfo info = null;
		for (WeightFormat ww : modelWeights) {
			String source = ww.getSourceFileName();
			if (!(new File(bmzModelFolder, source )).isFile() && !ww.getFramework().equals(ModelWeight.getTensorflowID()))
					continue;
			else if (ww.getFramework().equals(ModelWeight.getTensorflowID()) && !(new File(bmzModelFolder, source )).isFile()
					&& (!(new File(bmzModelFolder, "saved_model.pb" )).isFile() 
							|| !(new File(bmzModelFolder, "variables" )).isDirectory()))
				continue;
			info = EngineInfo.defineCompatibleDLEngineWithRdfYamlWeights(ww, enginesFolder);
			if (info != null) {
				modelSource = new File(bmzModelFolder, source).getAbsolutePath();
				break;
			}
		}
		if (info == null)
			throw new IOException("Please install a compatible engine with the model weights. "
					+ "To be compatible the engine has to be of the same framework and the major version needs to be the same. "
					+ "The model weights are: " + descriptor.getWeights().getSupportedWeightNamesAndVersion());
		BioimageIoModel model = new BioimageIoModel( info, bmzModelFolder, modelSource, classloader );
		model.descriptor = descriptor;
		model.tileCalculator = TileCalculator.init(descriptor);
		return model;
	}
	
	/**
	 * Load a model from the bioimage.io directly. Just providing the path to the
	 * folder where the rdf.yaml is, no extra info is needed as it is read from the
	 * rdf.yaml file
	 * To successfully create a Bioiamge.io model, it is required that there is installed
	 * at least one of the exact engines needed to load at least one of the weight formats
	 * in the exact version supported by the model. 
	 * Major and minor versions need to be the same (Tensorflow 2.7 != Tensorflow 2.4).
	 * 
	 * @param bmzModelFolder
	 * 	folder where the bioimage.io model is located (parent folder of the rdf.yaml file)
	 * @param enginesFolder
	 * 	directory where all the engine (DL framework) folders are downloaded
	 * @param classloader
	 * 	Parent ClassLoader of the engine (can be null). Almost the same method as 
	 *  Model.createBioimageioModel( String bmzModelFolder, String enginesFolder ). 
	 *  The only difference is that this method can choose the parent ClassLoader for the engine. 
	 *  JDLL creates a separate ChildFirst-ParentLast CustomClassLoader for each of the 
	 *  engines loaded to avoid conflicts between them. In order to have access to the 
	 *  classes of the main ClassLoader the ChildFirst-ParentLast CustomClassLoader needs a parent. 
	 *  If no classloader argument is provided the parent ClassLoader will be the Thread's 
	 *  context ClassLoader (Thread.currentThread().getContextClassLoader()).
	 *  The classloader argument can be null.
	 *  The classloader argument is usually not needed, but for some softwares 
	 *  such as Icy, that have a custom management of ClassLoaders it is necessary.
	 * @return a model ready to be loaded
	 * @throws LoadEngineException if there is any error loading the DL framework
	 * @throws IOException if there is any error finding the engines in the system
	 * @throws ModelSpecsException if the rdf.yaml file has some at least a field which does not comply with the Bioiamge.io constraints
	 * @throws IllegalStateException if any of the installed DL engines have been manipulated incorrectly
	 */
	public static BioimageIoModel createBioimageioModelWithExactWeigths(String bmzModelFolder, 
			String enginesFolder, ClassLoader classloader)
			throws IOException, ModelSpecsException, IllegalStateException, LoadEngineException {
		Objects.requireNonNull(bmzModelFolder);
		Objects.requireNonNull(enginesFolder);
		if (new File(bmzModelFolder, Constants.RDF_FNAME).isFile() == false)
			throw new IOException("A Bioimage.io model folder should contain its corresponding rdf.yaml file.");
		ModelDescriptor descriptor = 
				ModelDescriptorFactory.readFromLocalFile(bmzModelFolder + File.separator + Constants.RDF_FNAME);
		String modelSource = null;
		List<WeightFormat> modelWeights = descriptor.getWeights().gettAllSupportedWeightObjects();
		EngineInfo info = null;
		for (WeightFormat ww : modelWeights) {
			String source = ww.getSourceFileName();
			if (!(new File(bmzModelFolder, source )).isFile())
					continue;
			info = EngineInfo.defineExactDLEngineWithRdfYamlWeights(ww, enginesFolder);
			if (info != null) {
				modelSource = new File(bmzModelFolder, source).getAbsolutePath();
				break;
			}
		}
		if (info == null)
			throw new IOException("Please install the engines defined by the model weights. "
					+ "The model weights are: " + descriptor.getWeights().getSupportedWeightNamesAndVersion());
		BioimageIoModel model = new BioimageIoModel(info, bmzModelFolder, modelSource, classloader);
		model.descriptor = descriptor;
		model.tileCalculator = TileCalculator.init(descriptor);
		return model;
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
		if (!this.isLoaded())
			throw new RunModelException("Please first load the model.");
		if (!this.tiling) {
			List<Tensor<T>> outs = createOutputTensors();
			inference(inputTensors, outs);
			return outs;
		}
		List<ImageInfo> imageInfos = inputTensors.stream()
				.map(tt -> new ImageInfo(tt.getName(), tt.getAxesOrderString(), tt.getData().dimensionsAsLongArray()))
				.collect(Collectors.toList());
		List<TileInfo> inputTiles = tileCalculator.getOptimalTileSize(imageInfos);
		TileMaker maker = TileMaker.build(descriptor, inputTiles);
		List<Tensor<T>> outTensors = createOutputTensors(maker);
		return runBMZ(inputTensors, outTensors, maker);
	}
	
	private <T extends RealType<T> & NativeType<T>> List<Tensor<T>> createOutputTensors(TileMaker maker) {
		List<Tensor<T>> outputTensors = new ArrayList<Tensor<T>>();
		for (TensorSpec tt : descriptor.getOutputTensors()) {
			long[] dims = maker.getOutputImageSize(tt.getName());
			outputTensors.add((Tensor<T>) Tensor.buildBlankTensor(tt.getName(), 
																	tt.getAxesOrder(), 
																	dims, 
																	(T) new FloatType()));
		}
		return outputTensors;
	}
	
	private <T extends RealType<T> & NativeType<T>> List<Tensor<T>> createOutputTensors() {
		List<Tensor<T>> outputTensors = new ArrayList<Tensor<T>>();
		for (TensorSpec tt : descriptor.getOutputTensors()) {
			outputTensors.add(Tensor.buildEmptyTensor(tt.getName(), tt.getAxesOrder()));
		}
		return outputTensors;
	}
	
	private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> runBMZ(List<Tensor<R>> inputTensors, List<Tensor<T>> outputTensors, TileMaker tiles) throws RunModelException {
		Processing processing = Processing.init(descriptor);
		inputTensors = processing.preprocess(inputTensors, false);
		runTiling(inputTensors, outputTensors, tiles);
		return processing.postprocess(outputTensors, true);
	}

	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	void run(List<Tensor<T>> inputTensors, List<Tensor<R>> outputTensors) throws RunModelException {
		if (!this.isLoaded())
			throw new RunModelException("Please first load the model.");
		if (!this.tiling) {
			this.inference(inputTensors, outputTensors);
			return;
		}
		List<ImageInfo> imageInfos = inputTensors.stream()
				.map(tt -> new ImageInfo(tt.getName(), tt.getAxesOrderString(), tt.getData().dimensionsAsLongArray()))
				.collect(Collectors.toList());
		List<TileInfo> inputTiles = tileCalculator.getOptimalTileSize(imageInfos);
		TileMaker maker = TileMaker.build(descriptor, inputTiles);
		for (int i = 0; i < maker.getNumberOfTiles(); i ++) {
			Tensor<R> tt = outputTensors.get(i);
			long[] expectedSize = maker.getOutputImageSize(tt.getName());
			if (expectedSize == null) {
				throw new IllegalArgumentException("Tensor '" + tt.getName() + "' is missing in the outputs.");
			} else if (!tt.isEmpty() && Arrays.equals(expectedSize, tt.getData().dimensionsAsLongArray())) {
				throw new IllegalArgumentException("Tensor '" + tt.getName() + "' size is different than the expected size"
						+ " as defined by the rdf.yaml: " + Arrays.toString(tt.getData().dimensionsAsLongArray()) 
						+ " vs " + Arrays.toString(expectedSize) + ".");
			}
		}
		runBMZ(inputTensors, outputTensors, maker);
	}
	
	/**
	 * 
	 * @param <T>
	 * 	nothing
	 * @param args
	 * 	nothing
	 * @throws IOException	nothing
	 * @throws ModelSpecsException	nothing
	 * @throws LoadEngineException	nothing
	 * @throws RunModelException	nothing
	 * @throws LoadModelException	nothing
	 */
	public static <T extends NativeType<T> & RealType<T>> void main(String[] args) throws IOException, ModelSpecsException, LoadEngineException, RunModelException, LoadModelException {
		
		String mm = "/home/carlos/git/JDLL/models/NucleiSegmentationBoundaryModel_17122023_143125";
		Img<T> im = Cast.unchecked(ArrayImgs.floats(new long[] {1, 1, 512, 512}));
		List<Tensor<T>> l = new ArrayList<Tensor<T>>();
		l.add(Tensor.build("input0", "bcyx", im));
		BioimageIoModel model = createBioimageioModel(mm);
		model.loadModel();
		TileInfo tile = TileInfo.build(l.get(0).getName(), new long[] {1, 1, 512, 512}, 
				l.get(0).getAxesOrderString(), new long[] {1, 1, 512, 512}, l.get(0).getAxesOrderString());
		List<TileInfo> tileList = new ArrayList<TileInfo>();
		tileList.add(tile);
		model.run(l);
		System.out.println(false);
		
	}
	
	/**
	 * 
	 * @return whether the model is designed for the bioengine or not
	 */
	public boolean isBioengine() {
		return bioengine;
	}
	
	/**
	 * Get the {@link ModelDescriptor} instance that contains the specs defined in the 
	 * Bioimage.io rdf.yaml specs file.
	 * If the model does not contain a specs file, the methods returns null
	 * @return the {@link ModelDescriptor} instance that contains the specs defined in the 
	 * 	Bioimage.io rdf.yaml specs file.
	 * @throws IOException if any of the required files is corrupt or missing
	 * @throws ModelSpecsException if the specs of rdf.yaml file are not compliant with the Bioimage.io specs
	 * @throws FileNotFoundException if any of the files required is not found
	 */
	public ModelDescriptor getBioimageioSpecs() throws FileNotFoundException, ModelSpecsException, IOException {
		if (descriptor == null && new File(modelFolder + File.separator + Constants.RDF_FNAME).isFile()) {
			descriptor = ModelDescriptorFactory.readFromLocalFile(modelFolder + File.separator + Constants.RDF_FNAME);
		}
		return this.descriptor;
	}
}
