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

import java.io.Closeable;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.tiling.ImageInfo;
import io.bioimage.modelrunner.bioimageio.tiling.TileCalculator;
import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.bioimageio.bioengine.BioEngineAvailableModels;
import io.bioimage.modelrunner.bioimageio.bioengine.BioengineInterface;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.bioimageio.description.exceptions.ModelSpecsException;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat;
import io.bioimage.modelrunner.bioimageio.tiling.TileInfo;
import io.bioimage.modelrunner.bioimageio.tiling.TileMaker;
import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.engine.EngineLoader;
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
import net.imglib2.util.Util;

/**
 * Class that manages a Deep Learning model to load it and run it.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class Model implements Closeable
{
	/**
	 * Whether the model is loaded or not
	 */
	boolean loaded = false;
	/**
	 * ClassLoader containing all the classes needed to use the corresponding
	 * Deep Learning framework (engine).
	 */
	private EngineLoader engineClassLoader;

	/**
	 * All the information needed to load the engine corresponding to the model
	 * and the model itself.
	 */
	private EngineInfo engineInfo;

	/**
	 * Path to the folder containing the Bioimage.io model
	 */
	private String modelFolder;

	/**
	 * Source file of the Deep Learning model as defined in the yaml file
	 */
	private String modelSource;

	/**
	 * Model name as defined in the yaml file. For identification purposes
	 */
	private String modelName;
	/**
	 * Whether the model is created for the bioengine or not
	 */
	private boolean bioengine = false;
	/**
	 * Object containing the information of the rdf.yaml file of a Bioimage.io model
	 */
	private ModelDescriptor descriptor;

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
	private Model( EngineInfo engineInfo, String modelFolder, String modelSource, ClassLoader classLoader )
			throws LoadEngineException, MalformedURLException, IllegalStateException, IOException
	{
		if ( !engineInfo.isBioengine()
				&& !engineInfo.getFramework().equals(EngineInfo.getTensorflowKey())
				&& !engineInfo.getFramework().equals(EngineInfo.getBioimageioTfKey()) )
			Objects.requireNonNull(modelSource);
		this.engineInfo = engineInfo;
		this.modelFolder = modelFolder;
		this.modelSource = modelSource;
		setEngineClassLoader( classLoader );
	}

	/**
	 * Creates a DeepLearning model {@link Model} from the wanted Deep Learning
	 * framework (engine)
	 * 
	 * @param modelFolder
	 *            String path to the folder where all the components of the
	 *            model are stored
	 * @param modelSource
	 *            String path to the actual model file. In Pytorch is the path
	 *            to a .pt file and for Tf it is the same as the modelFolder
	 * @param engineInfo
	 *            all the information needed to load the classes of a Deep
	 *            Learning framework (engine)
	 * @return the Model that is going to be used to make inference
	 * @throws LoadEngineException
	 *             if there is an error finding the Deep LEarningn interface
	 *             that connects with the DL libraries
	 * @throws MalformedURLException if the JAR files are not well defined in the .json file
	 * @throws IOException if there is any error finding the engines in the system
	 * @throws IllegalStateException if any of the engines has been incorrectly modified
	 * @throws LoadEngineException if there is any error loading the engines
	 */
	public static Model createDeepLearningModel( String modelFolder, String modelSource, EngineInfo engineInfo )
			throws LoadEngineException, MalformedURLException, IllegalStateException, IOException
	{
		Objects.requireNonNull(modelFolder);
		Objects.requireNonNull(engineInfo);
		if ( !engineInfo.isBioengine() 
				&& !engineInfo.getFramework().equals(EngineInfo.getTensorflowKey())
				&& !engineInfo.getFramework().equals(EngineInfo.getBioimageioTfKey()) )
			Objects.requireNonNull(modelSource);
		Model model = new Model( engineInfo, modelFolder, modelSource, null );

		if (Paths.get(modelFolder, Constants.RDF_FNAME).toFile().isFile()) {
			try {
				model.descriptor = ModelDescriptorFactory.readFromLocalFile(Paths.get(modelFolder, Constants.RDF_FNAME).toAbsolutePath().toString());
			} catch (ModelSpecsException | IOException e) {
			}
		}
		return model;
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
	public static Model createBioimageioModel(String bmzModelFolder, ClassLoader classloader)
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
	public static Model createBioimageioModel(String bmzModelFolder)
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
	public static Model createBioimageioModel(String bmzModelFolder, String enginesFolder) 
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
	public static Model createBioimageioModel(String bmzModelFolder, String enginesFolder, ClassLoader classloader) 
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
		Model model = new Model( info, bmzModelFolder, modelSource, classloader );
		model.descriptor = descriptor;
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
	public static Model createBioimageioModelWithExactWeigths(String bmzModelFolder, 
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
		Model model = Model.createDeepLearningModel(bmzModelFolder, modelSource, info, classloader);
		model.descriptor = descriptor;
		return model;
	}

	/**
	 * Creates a DeepLearning model {@link Model} from the wanted Deep Learning
	 * framework (engine)
	 * 
	 * @param modelFolder
	 *            String path to the folder where all the components of the
	 *            model are stored
	 * @param modelSource
	 *            String path to the actual model file. In Pytorch is the path
	 *            to a .pt file and for Tf it is the same as the modelFolder
	 * @param engineInfo
	 *            all the information needed to load the classes of a Deep
	 *            Learning framework (engine)
	 * @param classLoader
	 * 	Parent ClassLoader of the engine(can be null). Almost the same method as 
	 *  Model.createDeepLearningModel( String modelFolder, String modelSource, EngineInfo engineInfo). 
	 *  The only difference is that this method can choose the parent ClassLoader for the engine. 
	 *  JDLL creates a separate ChildFirst-ParentLast CustomClassLoader for each of the 
	 *  engines loaded to avoid conflicts between them. In order to have access to the 
	 *  classes of the main ClassLoader the ChildFirst-ParentLast CustomClassLoader needs a parent. 
	 *  If no classloader argument is provided the parent ClassLoader will be the Thread's 
	 *  context ClassLoader (Thread.currentThread().getContextClassLoader()).
	 *  
	 *  The classloader argument is usually not needed, but for some softwares 
	 *  such as Icy, that have a custom management of ClassLoaders it is necessary.
	 * @return the Model that is going to be used to make inference
	 * @throws LoadEngineException if there is any error loading the DL framework
	 * @throws IOException if there is any error finding the engines in the system
	 * @throws IllegalStateException if any of the installed DL engines have been manipulated incorrectly
	 * @throws MalformedURLException if the JAR files are not well defined in the .json file
	 */
	public static Model createDeepLearningModel( String modelFolder, String modelSource, EngineInfo engineInfo,
			ClassLoader classLoader ) throws LoadEngineException, MalformedURLException, IllegalStateException, IOException
	{
		Objects.requireNonNull(modelFolder);
		Objects.requireNonNull(engineInfo);
		if ( !engineInfo.isBioengine()
				&& !engineInfo.getFramework().equals(EngineInfo.getTensorflowKey())
				&& !engineInfo.getFramework().equals(EngineInfo.getBioimageioTfKey()))
			Objects.requireNonNull(modelSource);
		Model model = new Model( engineInfo, modelFolder, modelSource, classLoader );

		if (Paths.get(modelFolder, Constants.RDF_FNAME).toFile().isFile()) {
			try {
				model.descriptor = ModelDescriptorFactory.readFromLocalFile(Paths.get(modelFolder, Constants.RDF_FNAME).toAbsolutePath().toString());
			} catch (ModelSpecsException | IOException e) {
			}
		}
		return model;
	}

	/**
	 * Load a model from the bioimage.io directly on the Bioengine. 
	 * Only the path to the model folder that contains the rdf.yaml is needed.
	 * To load a model on the bioengine we need to specify the server where our instance
	 * of the Bioengine is hosted.
	 * @param bmzModelFolder
	 * 	folder where the bioimage.io model is located (parent folder of the rdf.yaml file)
	 * @param serverURL
	 * 	url where the wanted insance of the bioengine is hosted
	 * @return a model ready to be loaded
	 * @throws Exception if there is any error creating the model (no rdf.yaml file,
	 *  or the url does not exist) or if the model is not supported on the Bioengine.
	 *  To check the models supported on the Bioengine, visit: https://raw.githubusercontent.com/bioimage-io/bioengine-model-runner/gh-pages/manifest.bioengine.yaml
	 */
	public static Model createBioimageioModelForBioengine(String bmzModelFolder, String serverURL) throws Exception {
		if (new File(bmzModelFolder, Constants.RDF_FNAME).isFile() == false)
			throw new IOException("A Bioimage.io model folder should contain its corresponding rdf.yaml file.");
		ModelDescriptor descriptor = 
				ModelDescriptorFactory.readFromLocalFile(bmzModelFolder + File.separator + Constants.RDF_FNAME);
		boolean valid = BioEngineAvailableModels.isModelSupportedInBioengine(descriptor.getModelID());
		if (!valid)
			throw new IllegalArgumentException("The selected model is currently not supported by the Bioegine. "
					+ "To check the list of supported models please visit: " + BioEngineAvailableModels.getBioengineJson());
		EngineInfo info = EngineInfo.defineBioengine(serverURL);
		Model model =  Model.createDeepLearningModel(bmzModelFolder, null, info);
		model.bioengine = true;
		model.descriptor = descriptor;
		return model;
	}

	/**
	 * Sets the classloader containing the Deep Learning engine
	 * 
	 * @param classLoader
	 *            parent ClassLoader of the engine (can be null)
	 * @throws LoadEngineException
	 *             if there is an error finding the Deep LEarningn interface
	 *             that connects with the DL libraries
	 * @throws MalformedURLException if the JAR files are not well defined in the .json file
	 * @throws IOException if there is any error finding the engines in the system
	 * @throws IllegalStateException if any of the engines has been incorrectly modified
	 */
	private void setEngineClassLoader( ClassLoader classLoader ) throws LoadEngineException, MalformedURLException, IllegalStateException, IOException
	{
		this.engineClassLoader = EngineLoader.createEngine(
				( classLoader == null ) ? Thread.currentThread().getContextClassLoader() : classLoader, engineInfo );
	}

	/**
	 * Load the model wanted to make inference into the particular ClassLoader
	 * created to run a specific Deep Learning framework (engine)
	 * 
	 * @throws LoadModelException
	 *             if the model was not loaded
	 */
	public void loadModel() throws LoadModelException
	{
		DeepLearningEngineInterface engineInstance = engineClassLoader.getEngineInstance();
		engineClassLoader.setEngineClassLoader();
		engineInstance.loadModel( modelFolder, modelSource );
		if (engineClassLoader.isBioengine())
			((BioengineInterface) engineInstance).addServer(engineInfo.getServer());
		engineClassLoader.setBaseClassLoader();
		loaded = true;
	}

	/**
	 * Close the Deep LEarning model in the ClassLoader where the Deep Learning
	 * framework has been called and instantiated
	 */
	@Override
	public void close()
	{
		if (getEngineClassLoader() == null)
			return;
		DeepLearningEngineInterface engineInstance = getEngineClassLoader().getEngineInstance();
		engineClassLoader.setEngineClassLoader();
		engineInstance.closeModel();
		getEngineClassLoader().close();
		engineInstance = null;
		engineClassLoader.setBaseClassLoader();
		engineClassLoader = null;
		loaded = false;
	}

	/**
	 * Method that calls the ClassLoader with the corresponding JARs of the Deep
	 * Learning framework (engine) loaded to run inference on the tensors. The
	 * method returns the corresponding output tensors
	 * 
	 * @param <T>
	 * 	ImgLib2 data type of the input tensors
	 * @param <R>
	 * 	ImgLib2 data type of the output tensors, it can be the same as in the input
	 * @param inTensors
	 *            input tensors containing all the tensor data
	 * @param outTensors
	 *            expected output tensors. Their backend data will be rewritten with the result of the inference
	 * @throws RunModelException
	 *             if the is any problem running the model
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	void runModel( List< Tensor < T > > inTensors, List< Tensor < R > > outTensors ) throws RunModelException
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
		engineInstance.run( inTensorsFloat, outTensors );
		engineClassLoader.setBaseClassLoader();
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
	 * @throws ModelSpecsException if the parameters of the rdf.yaml file are not correct
	 * @throws RunModelException if the model has not been previously loaded
	 * @throws IOException if any of the required files is missing or corrupt
	 * @throws FileNotFoundException if any of the required files is missing
	 * @throws IllegalArgumentException if the model is not a Bioimage.io model or if lacks a Bioimage.io
	 *  rdf.yaml specs file in the model folder. 
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> runBMZ(List<Tensor<R>> inputTensors) throws ModelSpecsException, RunModelException, FileNotFoundException, IOException {
		return runBMZ(inputTensors, new TilingConsumer());
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
	 * @param tileCounter
	 * 	consumer that counts the number of tiles processed out of the total, if null, nothing is counted
	 * @return the resulting tensors 
	 * @throws ModelSpecsException if the parameters of the rdf.yaml file are not correct
	 * @throws RunModelException if the model has not been previously loaded
	 * @throws IOException if any of the required files is missing or corrupt
	 * @throws FileNotFoundException if any of the required files is missing
	 * @throws IllegalArgumentException if the model is not a Bioimage.io model or if lacks a Bioimage.io
	 *  rdf.yaml specs file in the model folder. 
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> runBMZ(List<Tensor<R>> inputTensors, TilingConsumer tileCounter) throws ModelSpecsException, RunModelException, FileNotFoundException, IOException {
		if (!this.isLoaded())
			throw new RunModelException("Please first load the model.");
		if (descriptor == null && !(new File(modelFolder, Constants.RDF_FNAME).isFile()))
			throw new IllegalArgumentException("Automatic tiling can only be done if the model contains a Bioiamge.io rdf.yaml specs file.");
		else if (descriptor == null)
			descriptor = ModelDescriptorFactory.readFromLocalFile(modelFolder + File.separator + Constants.RDF_FNAME);
		TileCalculator calc = TileCalculator.init(descriptor);
		List<ImageInfo> imageInfos = inputTensors.stream()
				.map(tt -> new ImageInfo(tt.getName(), tt.getAxesOrderString(), tt.getData().dimensionsAsLongArray()))
				.collect(Collectors.toList());
		List<TileInfo> inputTiles = calc.getOptimalTileSize(imageInfos);
		TileMaker maker = TileMaker.build(descriptor, inputTiles);
		return runBMZ(inputTensors, maker, tileCounter);
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
	 * @param tiles
	 * 	List of {@link TileInfo} objects containing information about the image size and tile
	 * 	size of each of the input tensors to the model
	 * @return the resulting tensors 
	 * @throws ModelSpecsException if the parameters of the rdf.yaml file are not correct
	 * @throws RunModelException if the model has not been previously loaded
	 * @throws IllegalArgumentException if the model is not a Bioimage.io model or if lacks a Bioimage.io
	 *  rdf.yaml specs file in the model folder. 
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> runBMZ(List<Tensor<R>> inputTensors, 
			List<TileInfo> tiles) throws ModelSpecsException, RunModelException {
		return runBMZ(inputTensors, tiles, null);
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
	 * @param tiles
	 * 	List of {@link TileInfo} objects containing information about the image size and tile
	 * 	size of each of the input tensors to the model
	 * @param tileCounter
	 * 	consumer that counts the number of tiles processed out of the total, if null, nothing is counted
	 * @return the resulting tensors 
	 * @throws ModelSpecsException if the parameters of the rdf.yaml file are not correct
	 * @throws RunModelException if the model has not been previously loaded
	 * @throws IllegalArgumentException if the model is not a Bioimage.io model or if lacks a Bioimage.io
	 *  rdf.yaml specs file in the model folder. 
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> runBMZ(List<Tensor<R>> inputTensors, 
			List<TileInfo> tiles, TilingConsumer tileCounter) throws ModelSpecsException, RunModelException {
		
		if (!this.isLoaded())
			throw new RunModelException("Please first load the model.");
		if (descriptor == null && !(new File(modelFolder, Constants.RDF_FNAME).isFile()))
			throw new IllegalArgumentException("Automatic tiling can only be done if the model contains a Bioiamge.io rdf.yaml specs file.");
		else if (descriptor == null) {
			try {
				descriptor = ModelDescriptorFactory.readFromLocalFile(modelFolder + File.separator + Constants.RDF_FNAME);
			} catch (ModelSpecsException | IOException e) {
				throw new ModelSpecsException(Types.stackTrace(e));
			}
		}
		TileMaker maker = TileMaker.build(descriptor, tiles);
		return runBMZ(inputTensors, maker, tileCounter);
	}
	
	private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> runBMZ(List<Tensor<R>> inputTensors, TileMaker tiles, TilingConsumer tileCounter) throws RunModelException {
		Processing processing = Processing.init(descriptor);
		inputTensors = processing.preprocess(inputTensors, false);
		List<Tensor<R>> outputTensors = runTiling(inputTensors, tiles, tileCounter);
		return processing.postprocess(outputTensors, true);
	}
	
	@SuppressWarnings("unchecked")
	private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> runTiling(List<Tensor<R>> inputTensors, TileMaker tiles, TilingConsumer tileCounter) throws RunModelException {
		List<Tensor<T>> outputTensors = new ArrayList<Tensor<T>>();
		for (TensorSpec tt : descriptor.getOutputTensors()) {
			long[] dims = tiles.getOutputImageSize(tt.getName());
			outputTensors.add((Tensor<T>) Tensor.buildBlankTensor(tt.getName(), 
																	tt.getAxesOrder(), 
																	dims, 
																	(T) new FloatType()));
		}
		
		for (int i = 0; i < tiles.getNumberOfTiles(); i ++) {
			int nTile = 0 + i;
			List<Tensor<R>> inputTiles = inputTensors.stream()
					.map(tt -> tiles.getNthTileInput(tt, nTile)).collect(Collectors.toList());
			List<Tensor<T>> outputTiles = outputTensors.stream()
					.map(tt -> tiles.getNthTileOutput(tt, nTile)).collect(Collectors.toList());
			runModel(inputTiles, outputTiles);
		}
		return outputTensors;
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
		Model model = createBioimageioModel(mm);
		model.loadModel();
		TileInfo tile = TileInfo.build(l.get(0).getName(), new long[] {1, 1, 512, 512}, 
				l.get(0).getAxesOrderString(), new long[] {1, 1, 512, 512}, l.get(0).getAxesOrderString());
		List<TileInfo> tileList = new ArrayList<TileInfo>();
		tileList.add(tile);
		model.runBMZ(l, tileList);
		System.out.println(false);
		
	}

	/**
	 * Get the EngineClassLoader created by the DeepLearning Model
	 * {@link Model}. The EngineClassLoader loads the JAR files needed to use
	 * the corresponding Deep Learning framework (engine)
	 * 
	 * @return the Model corresponding EngineClassLoader
	 */
	public EngineLoader getEngineClassLoader()
	{
		return this.engineClassLoader;
	}

	/**
	 * Get the folder where this model is located
	 * 
	 * @return the folder where this model is located
	 */
	public String getModelFolder()
	{
		return this.modelFolder;
	}

	/**
	 * Get the source of this model as specified in the yaml file
	 * 
	 * @return the source of this model from the yaml file
	 */
	public String getModelSource()
	{
		return this.modelSource;
	}

	/**
	 * Gets the name of the model
	 * 
	 * @return the name of the model
	 */
	public String getModelName()
	{
		return this.modelName;
	}
	
	/**
	 * 
	 * @return whether the model is designed for the bioengine or not
	 */
	public boolean isBioengine() {
		return bioengine;
	}
	
	/**
	 * Add method to get the {@link EngineInfo} used to create the model
	 * @return the {@link EngineInfo} used to create the model
	 */
	public EngineInfo getEngineInfo() {
		return engineInfo;
	}
	
	/**
	 * Whether the model is loaded or not
	 * @return whether the model is loaded or not
	 */
	public boolean isLoaded() {
		return loaded;
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
	
	/**
	 * Create consumer used to be used with {@link Model} for the methods {@link #runBMZ(List, TilingConsumer)}
	 * or {@link #runBMZ(List, List, TilingConsumer)}.
	 * The consumer helps to track the number if tiles that have already been processed.
	 * @return a consumer to track the tiling process
	 */
	public static TilingConsumer createTilingConsumer() {
		return new TilingConsumer();
	}
	
	/**
	 * Functional interface to create a consumer that is able to keep track of how many 
	 * tiles have been processed out if the total needed
	 * @author Carlos Garcia Lopez de Haro
	 */
	public static class TilingConsumer {
		/**
		 * Total tiles needed to process
		 */
		private Long totalTiles;
		/**
		 * Already processed tiles
		 */
		private Long tilesProcessed;
		
		/**
		 * Set the total numer of tiles that need to be processed
		 * @param totalTiles
		 * 	Total tiles that need to be processed
		 */
	    public void acceptTotal(Long totalTiles) {
	        this.totalTiles = totalTiles;
	    }
	    
	    /**
	     * Set the current number of tiles that have already been processed
	     * @param tilesProcessed
	     * 	The current number of tiles that have already been processed
	     */
	    public void acceptProgress(Long tilesProcessed) {
	        this.tilesProcessed = tilesProcessed;
	    }
	    
	    /**
	     * Get the total number of tiles that need to be processed
	     * @return the total number of tiles that need to be processed
	     */
	    public Long getTotalTiles() {
	    	return totalTiles;
	    }
	    
	    /**
	     * Get the number of tiles that have already been processed
	     * @return the number of tiles that have already been processed
	     */
	    public Long getTilesProcessed() {
	    	return tilesProcessed;
	    }
	}
}
