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
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.bioengine.BioengineInterface;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.bioimageio.description.exceptions.ModelSpecsException;
import io.bioimage.modelrunner.bioimageio.tiling.TileMaker;
import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.engine.EngineLoader;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.versionmanagement.InstalledEngines;
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
public class DLModel extends BaseModel
{
	/**
	 * Whether the model is loaded or not
	 */
	protected boolean loaded = false;
	/**
	 * ClassLoader containing all the classes needed to use the corresponding
	 * Deep Learning framework (engine).
	 */
	protected EngineLoader engineClassLoader;

	/**
	 * All the information needed to load the engine corresponding to the model
	 * and the model itself.
	 */
	protected EngineInfo engineInfo;

	/**
	 * Path to the folder containing the Bioimage.io model
	 */
	protected String modelFolder;

	/**
	 * Source file of the Deep Learning model as defined in the yaml file
	 */
	protected String modelSource;

	/**
	 * Model name as defined in the yaml file. For identification purposes
	 */
	protected String modelName;

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
	protected DLModel(EngineInfo engineInfo, String modelFolder, String modelSource, ClassLoader classLoader )
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

	@Override
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

	@Override
	/**
	 * Close the Deep LEarning model in the ClassLoader where the Deep Learning
	 * framework has been called and instantiated
	 */
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

	@Override

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
	void run( List< Tensor < T > > inTensors, List< Tensor < R > > outTensors ) throws RunModelException
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

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> List<Tensor<T>> run(
			List<Tensor<R>> inputTensors) throws RunModelException {
		// TODO Auto-generated method stub
		return null;
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
	 * Creates a DeepLearning model {@link BioimageIoModel} from the wanted Deep Learning
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
	public static DLModel createDeepLearningModel( String modelFolder, String modelSource, EngineInfo engineInfo )
			throws LoadEngineException, MalformedURLException, IllegalStateException, IOException
	{
		Objects.requireNonNull(modelFolder);
		Objects.requireNonNull(engineInfo);
		if ( !engineInfo.isBioengine() 
				&& !engineInfo.getFramework().equals(EngineInfo.getTensorflowKey())
				&& !engineInfo.getFramework().equals(EngineInfo.getBioimageioTfKey()) )
			Objects.requireNonNull(modelSource);

		if (Paths.get(modelFolder, Constants.RDF_FNAME).toFile().isFile()) {
			try {
				BioimageIoModel model = new BioimageIoModel( engineInfo, modelFolder, modelSource, null );
				model.descriptor = ModelDescriptorFactory.readFromLocalFile(Paths.get(modelFolder, Constants.RDF_FNAME).toAbsolutePath().toString());
				return model;
			} catch (ModelSpecsException | IOException e) {
				return new DLModel( engineInfo, modelFolder, modelSource, null );
			}
		}
		return new DLModel( engineInfo, modelFolder, modelSource, null );
	}

	/**
	 * Creates a DeepLearning model {@link BioimageIoModel} from the wanted Deep Learning
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
	public static BioimageIoModel createDeepLearningModel( String modelFolder, String modelSource, EngineInfo engineInfo,
			ClassLoader classLoader ) throws LoadEngineException, MalformedURLException, IllegalStateException, IOException
	{
		Objects.requireNonNull(modelFolder);
		Objects.requireNonNull(engineInfo);
		if ( !engineInfo.isBioengine()
				&& !engineInfo.getFramework().equals(EngineInfo.getTensorflowKey())
				&& !engineInfo.getFramework().equals(EngineInfo.getBioimageioTfKey()))
			Objects.requireNonNull(modelSource);
		BioimageIoModel model = new BioimageIoModel( engineInfo, modelFolder, modelSource, classLoader );

		if (Paths.get(modelFolder, Constants.RDF_FNAME).toFile().isFile()) {
			try {
				model.descriptor = ModelDescriptorFactory.readFromLocalFile(Paths.get(modelFolder, Constants.RDF_FNAME).toAbsolutePath().toString());
			} catch (ModelSpecsException | IOException e) {
			}
		}
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
	protected void setEngineClassLoader( ClassLoader classLoader ) throws LoadEngineException, MalformedURLException, IllegalStateException, IOException
	{
		this.engineClassLoader = EngineLoader.createEngine(
				( classLoader == null ) ? Thread.currentThread().getContextClassLoader() : classLoader, engineInfo );
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
	 * Get the EngineClassLoader created by the DeepLearning Model
	 * {@link BioimageIoModel}. The EngineClassLoader loads the JAR files needed to use
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
	 * Create consumer used to be used with {@link BioimageIoModel} for the methods {@link #runBMZ(List, TilingConsumer)}
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
