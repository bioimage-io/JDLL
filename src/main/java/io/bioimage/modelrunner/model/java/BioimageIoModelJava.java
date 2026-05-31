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
/**
 * 
 */
package io.bioimage.modelrunner.model.java;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat;
import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.processing.Processing;
import io.bioimage.modelrunner.model.tiling.ImageInfo;
import io.bioimage.modelrunner.model.tiling.TileCalculator;
import io.bioimage.modelrunner.model.tiling.TileInfo;
import io.bioimage.modelrunner.model.tiling.TileMaker;
import io.bioimage.modelrunner.model.tiling.merger.DenseMerger;
import io.bioimage.modelrunner.model.tiling.merger.Merger;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.versionmanagement.InstalledEngines;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;

/**
 * Class that manages a Deep Learning model to load it and run it.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class BioimageIoModelJava extends DLModelJava
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
	@SuppressWarnings("rawtypes")
	private DenseMerger merger;

	/**
	 * Construct the object model with all the needed information to load a
	 * model and make inference
	 *
	 * @param engineInfo the engineInfo parameter.
	 * @param modelFolder the modelFolder parameter.
	 * @param modelSource the modelSource parameter.
	 * @param classLoader the classLoader parameter.
	 * @throws LoadEngineException if a LoadEngineException occurs while executing this method.
	 * @throws MalformedURLException if a MalformedURLException occurs while executing this method.
	 * @throws IllegalStateException if a IllegalStateException occurs while executing this method.
	 * @throws IOException if an I/O error occurs.
	 */
	protected BioimageIoModelJava( EngineInfo engineInfo, String modelFolder, String modelSource, ClassLoader classLoader )
			throws LoadEngineException, MalformedURLException, IllegalStateException, IOException
	{
		super(engineInfo, modelFolder, modelSource, classLoader);
		this.tiling = true;
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
	 */
	public static BioimageIoModelJava createBioimageioModel(String bmzModelFolder, ClassLoader classloader)
			throws LoadEngineException, IOException {
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
	 */
	public static BioimageIoModelJava createBioimageioModel(String bmzModelFolder)
			throws LoadEngineException, IOException {
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
	 */
	public static BioimageIoModelJava createBioimageioModel(String bmzModelFolder, String enginesFolder) 
			throws LoadEngineException, IOException {
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
	 */
	public static BioimageIoModelJava createBioimageioModel(String bmzModelFolder, String enginesFolder, ClassLoader classloader) 
			throws LoadEngineException, IOException {
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
		BioimageIoModelJava model = new BioimageIoModelJava( info, bmzModelFolder, modelSource, classloader );
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
	 * @param bmzModelFolder the bmzModelFolder parameter.
	 * @param enginesFolder the enginesFolder parameter.
	 * @param classloader the classloader parameter.
	 * @return the resulting value.
	 * @throws IOException if an I/O error occurs.
	 * @throws IllegalStateException if a IllegalStateException occurs while executing this method.
	 * @throws LoadEngineException if a LoadEngineException occurs while executing this method.
	 */
	public static BioimageIoModelJava createBioimageioModelWithExactWeigths(String bmzModelFolder, 
			String enginesFolder, ClassLoader classloader)
			throws IOException, IllegalStateException, LoadEngineException {
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
		BioimageIoModelJava model = new BioimageIoModelJava(info, bmzModelFolder, modelSource, classloader);
		model.descriptor = descriptor;
		model.tileCalculator = TileCalculator.init(descriptor);
		return model;
	}
	protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List< Tensor < R > > backboneSingleInferenceTile( List< Tensor < T > > inTensors) throws RunModelException
	{
		DeepLearningEngineInterface engineInstance = engineClassLoader.getEngineInstance();
		engineClassLoader.setEngineClassLoader();
		ArrayList<Tensor<FloatType>> inTensorsFloat = new ArrayList<Tensor<FloatType>>();
		for (Tensor<T> tt : inTensors) {
			if (Util.getTypeFromInterval(tt.getData()) instanceof FloatType)
				inTensorsFloat.add(Cast.unchecked(tt));
			else
				inTensorsFloat.add(Tensor.createCopyOfTensorInWantedDataType( tt, new FloatType() ));
		}
		List<Tensor<R>> outTensors = getOutputTile(getTilingCounter().getTilesProcessed().intValue());
		engineInstance.run(inTensorsFloat, outTensors);
		engineClassLoader.setBaseClassLoader();
		return outTensors;
	}

	@SuppressWarnings("unchecked")
	private <R extends RealType<R> & NativeType<R>> List<Tensor<R>> getOutputTile(int patchNumber) {
		if (this.merger == null)
			throw new IllegalStateException("Bioimage.io Java output merger has not been configured.");
		return (List<Tensor<R>>) this.merger.getOutput(patchNumber);
	}

	@Override
	protected String getOutputTensorAxes(int outputCount) {
		if (descriptor.getOutputTensors().size() <= outputCount)
			throw new IllegalArgumentException("Cellpose only has 6 outputs.");
		return this.descriptor.getOutputTensors().get(outputCount).getAxesOrder();
	}

	@Override
	protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	Merger<Tensor<T>, Tensor<R>> getTileMaker(final List<Tensor<T>> inputs) {

		List<ImageInfo> imageInfos = inputs.stream()
				.map(tt -> new ImageInfo(tt.getName(), tt.getAxesOrderString(), tt.getData().dimensionsAsLongArray()))
				.collect(Collectors.toList());
		this.tileCalculator = TileCalculator.init(descriptor);
		List<TileInfo> inputTiles = tileCalculator.getOptimalTileSize(imageInfos);
		TileMaker tileMaker = TileMaker.build(descriptor, inputTiles);		
		DenseMerger<T, R> merger = new DenseMerger<T, R>(tileMaker);
		this.merger = merger;

		Processing processing = Processing.init(descriptor);
		processing.preprocess(inputs, true);		
		
		merger.addCallback(reconstructed -> processing.postprocess(reconstructed, true));
		merger.configure(inputs);
		return merger;
	}
	
	/**
	 * Executes main.
	 *
	 * @param args the args parameter.
	 * @throws IOException if an I/O error occurs.
	 * @throws LoadEngineException if a LoadEngineException occurs while executing this method.
	 * @throws RunModelException if a RunModelException occurs while executing this method.
	 * @throws LoadModelException if a LoadModelException occurs while executing this method.
	 */
	public static <T extends NativeType<T> & RealType<T>> void main(String[] args) throws IOException, LoadEngineException, RunModelException, LoadModelException {
		
		String mm = "/home/carlos/git/JDLL/models/NucleiSegmentationBoundaryModel_17122023_143125";
		Img<T> im = Cast.unchecked(ArrayImgs.floats(new long[] {1, 1, 512, 512}));
		List<Tensor<T>> l = new ArrayList<Tensor<T>>();
		l.add(Tensor.build("input0", "bcyx", im));
		BioimageIoModelJava model = createBioimageioModel(mm);
		model.loadModel();
		TileInfo tile = TileInfo.build(l.get(0).getName(), new long[] {1, 1, 512, 512}, 
				l.get(0).getAxesOrderString(), new long[] {1, 1, 512, 512}, l.get(0).getAxesOrderString());
		List<TileInfo> tileList = new ArrayList<TileInfo>();
		tileList.add(tile);
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
	 *
	 * @return the resulting value.
	 * @throws FileNotFoundException if a FileNotFoundException occurs while executing this method.
	 * @throws IOException if an I/O error occurs.
	 */
	public ModelDescriptor getBioimageioSpecs() throws FileNotFoundException, IOException {
		if (descriptor == null && new File(modelFolder + File.separator + Constants.RDF_FNAME).isFile()) {
			descriptor = ModelDescriptorFactory.readFromLocalFile(modelFolder + File.separator + Constants.RDF_FNAME);
		}
		return this.descriptor;
	}
}
