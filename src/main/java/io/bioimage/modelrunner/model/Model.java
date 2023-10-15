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
/**
 * 
 */
package io.bioimage.modelrunner.model;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.List;
import java.util.Objects;

import javax.xml.bind.ValidationException;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat;
import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.engine.EngineLoader;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tiling.PatchGridCalculator;
import io.bioimage.modelrunner.tiling.PatchSpec;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.versionmanagement.InstalledEngines;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * Class that manages a Deep Learning model to load it and run it.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class Model
{
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
		return new Model( engineInfo, modelFolder, modelSource, null );
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
	 * @throws ValidationException if the rdf.yaml file has some at least a field which does not comply with the Bioiamge.io constraints
	 */
	public static Model createBioimageioModel(String bmzModelFolder, ClassLoader classloader)
			throws LoadEngineException, ValidationException, IOException {
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
	 * @throws ValidationException if the rdf.yaml file has some at least a field which does not comply with the Bioiamge.io constraints
	 */
	public static Model createBioimageioModel(String bmzModelFolder)
			throws ValidationException, LoadEngineException, IOException {
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
	 * @throws ValidationException if the rdf.yaml file has some at least a field which does not comply with the Bioiamge.io constraints
	 */
	public static Model createBioimageioModel(String bmzModelFolder, String enginesFolder) 
			throws ValidationException, LoadEngineException, IOException {
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
	 * @throws ValidationException if the rdf.yaml file has some at least a field which does not comply with the Bioiamge.io constraints
	 */
	public static Model createBioimageioModel(String bmzModelFolder, String enginesFolder, ClassLoader classloader) 
			throws LoadEngineException, IOException, ValidationException {
		Objects.requireNonNull(bmzModelFolder);
		Objects.requireNonNull(enginesFolder);
		if (new File(bmzModelFolder, Constants.RDF_FNAME).isFile() == false)
			throw new IOException("A Bioimage.io model folder should contain its corresponding rdf.yaml file.");
		ModelDescriptor descriptor = 
			ModelDescriptor.readFromLocalFile(bmzModelFolder + File.separator + Constants.RDF_FNAME, false);
		String modelSource = null;
		List<WeightFormat> modelWeights = descriptor.getWeights().getSupportedWeights();
		EngineInfo info = null;
		for (WeightFormat ww : modelWeights) {
			String source = ww.getSource();
			if (!(new File(bmzModelFolder, source.substring(source.lastIndexOf("/")) )).isFile())
					continue;
			info = EngineInfo.defineCompatibleDLEngineWithRdfYamlWeights(ww, enginesFolder);
			if (info != null) {
				modelSource = new File(bmzModelFolder, 
						source.substring(source.lastIndexOf("/"))).getAbsolutePath();
				break;
			}
		}
		if (info == null)
			throw new IOException("Please install a compatible engine with the model weights. "
					+ "To be compatible the engine has to be of the same framework and the major version needs to be the same. "
					+ "The model weights are: " + descriptor.getWeights().getEnginesListWithVersions());
		return Model.createDeepLearningModel(bmzModelFolder, modelSource, info);
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
	 * @return a model ready to be loaded
	 * @throws LoadEngineException if there is any error loading the DL framework
	 * @throws IOException if there is any error finding the engines in the system
	 * @throws ValidationException if the rdf.yaml file has some at least a field which does not comply with the Bioiamge.io constraints
	 * @throws IllegalStateException if any of the installed DL engines have been manipulated incorrectly
	 */
	public static Model createBioimageioModelWithExactWeigths(String bmzModelFolder, String enginesFolder)
			throws IOException, ValidationException, IllegalStateException, LoadEngineException {
		Objects.requireNonNull(bmzModelFolder);
		Objects.requireNonNull(enginesFolder);
		if (new File(bmzModelFolder, Constants.RDF_FNAME).isFile() == false)
			throw new IOException("A Bioimage.io model folder should contain its corresponding rdf.yaml file.");
		ModelDescriptor descriptor = 
			ModelDescriptor.readFromLocalFile(bmzModelFolder + File.separator + Constants.RDF_FNAME, false);
		String modelSource = null;
		List<WeightFormat> modelWeights = descriptor.getWeights().getSupportedWeights();
		EngineInfo info = null;
		for (WeightFormat ww : modelWeights) {
			String source = ww.getSource();
			if (!(new File(bmzModelFolder, source.substring(source.lastIndexOf("/")) )).isFile())
					continue;
			info = EngineInfo.defineExactDLEngineWithRdfYamlWeights(ww, enginesFolder);
			if (info != null) {
				modelSource = new File(bmzModelFolder, 
						source.substring(source.lastIndexOf("/"))).getAbsolutePath();
				break;
			}
		}
		if (info == null)
			throw new IOException("Please install the engines defined by the model weights. "
					+ "The model weights are: " + descriptor.getWeights().getEnginesListWithVersions());
		return Model.createDeepLearningModel(bmzModelFolder, modelSource, info);
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
	 * @throws ValidationException if the rdf.yaml file has some at least a field which does not comply with the Bioiamge.io constraints
	 * @throws IllegalStateException if any of the installed DL engines have been manipulated incorrectly
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
		return new Model( engineInfo, modelFolder, modelSource, classLoader );
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
		engineClassLoader.setBaseClassLoader();
	}

	/**
	 * Close the Deep LEarning model in the ClassLoader where the Deep Learning
	 * framework has been called and instantiated
	 */
	public void closeModel()
	{
		DeepLearningEngineInterface engineInstance = getEngineClassLoader().getEngineInstance();
		engineClassLoader.setEngineClassLoader();
		engineInstance.closeModel();
		getEngineClassLoader().close();
		engineInstance = null;
		engineClassLoader.setBaseClassLoader();
		engineClassLoader = null;
	}

	/**
	 * Method that calls the ClassLoader with the corresponding JARs of the Deep
	 * Learning framework (engine) loaded to run inference on the tensors. The
	 * method returns the corresponding output tensors
	 * 
	 * @param inTensors
	 *            input tensors containing all the tensor data
	 * @param outTensors
	 *            expected output tensors. Their backend data will be rewritten with the result of the inference
	 * @throws RunModelException
	 *             if the is any problem running the model
	 */
	public void runModel( List< Tensor < ? > > inTensors, List< Tensor < ? > > outTensors ) throws RunModelException
	{
		DeepLearningEngineInterface engineInstance = engineClassLoader.getEngineInstance();
		engineClassLoader.setEngineClassLoader();
		inTensors.stream().forEach( tt -> tt = Tensor.createCopyOfTensorInWantedDataType( tt, new FloatType() ) );
		engineInstance.run( inTensors, outTensors );
		engineClassLoader.setBaseClassLoader();
	}
	
	/**
	 * 
	 * @param <T>
	 * 	ImgLib2 data type of the output images
	 * @param <R>
	 * @param inputImgs
	 * @return
	 * @throws ValidationException 
	 * @throws Exception 
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Img<T>> runBioimageioModelOnImgLib2WithTiling(List<RandomAccessibleInterval<R>> inputImgs) throws ValidationException {
		if (descriptor == null && modelFolder == null)
			throw new IllegalArgumentException("");
		else if (descriptor == null && !(new File(modelFolder, Constants.RDF_FNAME).isFile()))
			throw new IllegalArgumentException("");
		else if (descriptor == null)
			descriptor = ModelDescriptor.readFromLocalFile(modelFolder + File.separator + Constants.RDF_FNAME);
		PatchGridCalculator tileGrid = PatchGridCalculator.build(descriptor, inputImgs);
		List<PatchSpec> specs = tileGrid.call();
		return null;
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
}
