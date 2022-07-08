/**
 * 
 */
package org.bioimageanalysis.icy.deeplearning;

import java.util.List;

import org.bioimageanalysis.icy.deeplearning.exceptions.LoadEngineException;
import org.bioimageanalysis.icy.deeplearning.exceptions.LoadModelException;
import org.bioimageanalysis.icy.deeplearning.exceptions.RunModelException;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bioimageanalysis.icy.deeplearning.utils.DeepLearningInterface;
import org.bioimageanalysis.icy.deeplearning.utils.EngineInfo;
import org.bioimageanalysis.icy.deeplearning.utils.EngineLoader;

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
     * Source file of the Deep Learning model as defined in the
     * yaml file
     */
    private String modelSource;
    /**
     * Model name as defined in the yaml file. For identification purposes
     */
    private String modelName;

    /**
     * Construct the object model with all the needed information to
     * load a model and make inference
     * 
     * @param engineInfo
     *        informaton needed about the model
     * @param modelFolder
     *        directory where of the model folder
     * @param modelSource
     *        name of the actual model file (.pt for torchscript)
     * @param classLoader
     *        parent ClassLoader (can be null)
     * @throws LoadEngineException
     *         if there is an error finding the Deep LEarningn interface
     *         that connects with the DL libraries
     * @throws Exception
     *         if the directory is not found
     */
    private Model(EngineInfo engineInfo, String modelFolder, String modelSource, ClassLoader classLoader) throws LoadEngineException, Exception
    {
        this.engineInfo = engineInfo;
        this.modelFolder = modelFolder;
        this.modelSource = modelSource;
        setEngineClassLoader(classLoader);
    }

    /**
     * Creates a DeepLearning model {@link Model} from the wanted Deep Learning
     * framework (engine)
     * 
     * @param modelFolder
     *        String path to the folder where all the components of the model are stored
     * @param modelSource
     *        String path to the actual model file. In Pytorch is the path to a .pt file
     *        and for Tf it is the same as the modelFolder
     * @param engineInfo
     *        all the information needed to load the classes of
     *        a Deep Learning framework (engine)
     * @return the Model that is going to be used to make inference
     * @throws LoadEngineException
     *         if there is an error finding the Deep LEarningn interface
     *         that connects with the DL libraries
     * @throws Exception
     *         if the directory is not found
     */
    public static Model createDeepLearningModel(String modelFolder, String modelSource, EngineInfo engineInfo)
            throws LoadEngineException, Exception
    {
        return new Model(engineInfo, modelFolder, modelSource, null);
    }

    /**
     * Creates a DeepLearning model {@link Model} from the wanted Deep Learning
     * framework (engine)
     * 
     * @param modelFolder
     *        String path to the folder where all the components of the model are stored
     * @param modelSource
     *        String path to the actual model file. In Pytorch is the path to a .pt file
     *        and for Tf it is the same as the modelFolder
     * @param engineInfo
     *        all the information needed to load the classes of
     *        a Deep Learning framework (engine)
     * @param classLoader
     *        parent ClassLoader (can be null)
     * @return the Model that is going to be used to make inference
     * @throws LoadEngineException
     *         if there is an error finding the Deep LEarningn interface
     *         that connects with the DL libraries
     * @throws Exception
     *         if the directory is not found
     */
    public static Model createDeepLearningModel(String modelFolder, String modelSource, EngineInfo engineInfo, ClassLoader classLoader)
            throws LoadEngineException, Exception
    {
        return new Model(engineInfo, modelFolder, modelSource, classLoader);
    }

    /**
     * Sets the classloader containing the Deep Learning engine
     * 
     * @param classLoader
     *        parent ClassLoader (can be null)
     * @throws LoadEngineException
     *         if there is an error finding the Deep LEarningn interface
     *         that connects with the DL libraries
     * @throws Exception
     *         if the directory is not found
     */
    public void setEngineClassLoader(ClassLoader classLoader) throws LoadEngineException, Exception
    {
        this.engineClassLoader = EngineLoader.createEngine((classLoader == null) ? Thread.currentThread().getContextClassLoader() : classLoader, engineInfo);
    }

    /**
     * Load the model wanted to make inference into the particular ClassLoader
     * created to run a specific Deep Learning framework (engine)
     * 
     * @throws LoadModelException
     *         if the model was not loaded
     */
    public void loadModel() throws LoadModelException
    {
        DeepLearningInterface engineInstance = engineClassLoader.getEngineInstance();
        engineClassLoader.setEngineClassLoader();
        engineInstance.loadModel(modelFolder, modelSource);
        engineClassLoader.setIcyClassLoader();
    }

    /**
     * Close the Deep LEarning model in the ClassLoader where the Deep Learning
     * framework has been called and instantiated
     */
    public void closeModel()
    {
        DeepLearningInterface engineInstance = getEngineClassLoader().getEngineInstance();
        engineClassLoader.setEngineClassLoader();
        engineInstance.closeModel();
        getEngineClassLoader().close();
        engineInstance = null;
        engineClassLoader.setIcyClassLoader();
        engineClassLoader = null;
    }

    /**
     * Method that calls the ClassLoader with the corresponding JARs of the
     * Deep Learning framework (engine) loaded to run inference on the tensors.
     * The method returns the corresponding output tensors
     * 
     * @param inTensors
     *        input tensors containing all the tensor data
     * @param outTensors
     *        output tensors expected containing only the axes order and names. The
     *        data will be filled with the outputs of the models
     * @return the output tensors produced by the model
     * @throws RunModelException
     *         if the is any problem running the model
     * @throws RunModelException
     *         if there is any problem closing the tensors
     */
    public List<Tensor> runModel(List<Tensor> inTensors, List<Tensor> outTensors) throws RunModelException, Exception
    {
        DeepLearningInterface engineInstance = engineClassLoader.getEngineInstance();
        engineClassLoader.setEngineClassLoader();
        inTensors.stream().forEach(tt -> tt.convertToDataType(new FloatType()));
        outTensors = engineInstance.run(inTensors, outTensors);
        engineClassLoader.setIcyClassLoader();
        return outTensors;
    }

    /**
     * Get the EngineClassLoader created by the DeepLearning Model {@link Model}.
     * The EngineClassLoader loads the JAR files needed to use the corresponding
     * Deep Learning framework (engine)
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
     * Sets the model name
     * 
     * @param modelName
     *        the name of the model
     */
    public void setModelName(String modelName)
    {
        this.modelName = modelName;
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

}
