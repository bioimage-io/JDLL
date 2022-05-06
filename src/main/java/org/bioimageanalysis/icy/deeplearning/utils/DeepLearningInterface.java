package org.bioimageanalysis.icy.deeplearning.utils;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bioimageanalysis.icy.deeplearning.tensor.TensorAPIManager;
import org.bioimageanalysis.icy.deeplearning.tensor.TensorManager;

import ai.djl.ndarray.NDManager;

import org.bioimageanalysis.icy.deeplearning.exceptions.LoadModelException;
import org.bioimageanalysis.icy.deeplearning.exceptions.RunModelException;

public interface DeepLearningInterface {
	

	
	/**
	 * Load the NDArray neeeded dependencies into memory to load the needed engines and native libraries
	 * @throws ClassNotFoundException if the NDManager class is not found
	 * @throws NoSuchMethodException is the methods to create a NDManager are not found
	 * @throws SecurityException if there is a security exception
	 * @throws IllegalAccessException if there is an illegal access
	 * @throws IllegalArgumentException if one of the argumnets does not correspond
	 * @throws InvocationTargetException if there is any error invoking the NDManager
	 */
	default void initializeNDArrays() throws ClassNotFoundException, NoSuchMethodException, SecurityException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		ClassLoader engineClassloader = Thread.currentThread().getContextClassLoader();
		Class<?> cl = engineClassloader.loadClass("ai.djl.ndarray.NDManager");
	    Method m = cl.getMethod("newBaseManager");
	    Object manager = m.invoke(null);
	    ((NDManager) manager).close();
	    manager = null;
	}
	
	/**
	 * Default method to run an external Deep Learning framework (engine). It converts first converts
	 * the tensor lists from API version agnostic Buffer based tensors into API version dependent
	 * NDArray based tensors.
	 * @param inputTensors
	 * 	list containing the input tensors
	 * @param outputTensors
	 * 	list containing only the information about output tensors
	 * @return
	 * 	output tensors produced by the model
	 * @throws RunModelException if there is an error in the execution of the model
	 * @throws Exception if there is an error closing the tensors or tensormanager
	 */
	default List<Tensor> runEngine(List<Tensor> inputTensors, List<Tensor> outputTensors) throws RunModelException, Exception {
		// Convert the lists of tensors, which have to be using Buffers as backend, into
		// a list of tensors using the corresponding API version NDArrays as backend
		List<Tensor> copyInputTensors = TensorAPIManager.createTensorsCopyIntoAPI(inputTensors);
		TensorManager manager = copyInputTensors.get(0).getManager();
		List<Tensor> copyOutputTensors = TensorAPIManager.createTensorsCopyIntoAPI(outputTensors, manager);
		copyOutputTensors = run(copyInputTensors, copyOutputTensors);
		TensorAPIManager.tensorsAsBuffers(copyInputTensors);
		TensorAPIManager.tensorsAsBuffers(copyOutputTensors);
		TensorAPIManager.copyTensorsIntoAPIAsBuffers(copyInputTensors, inputTensors);
		TensorAPIManager.copyTensorsIntoAPIAsBuffers(copyOutputTensors, outputTensors);
		manager.getManager().close();
		copyInputTensors.stream().forEach(tensor -> tensor.close());
		copyOutputTensors.stream().forEach(tensor -> tensor.close());
		return outputTensors;
		
	}
	
	/**
	 * Method that the interface implements to make inference.
	 * In the class that implements the interface, the code to run
	 * the model on the tensors should go here.
	 * 
	 * @param inputTensors
	 * 	list containing the input tensors
	 * @param outputTensors
	 * 	list containing only the information about output tensors
	 * @return
	 * 	output tensors produced by the model
	 * @throws Exception if there is an error in the execution of the model
	 */
    public List<Tensor> run(List<Tensor> inputTensors, List<Tensor> outputTensors) throws RunModelException;
    
    /**
     * Load the model with the corresponding engine on the particular
     * independent ClassLoader. This is done to be able to load the model
     * only one time and use it several times.
     * 
	 * @param modelFolder
	 * 	String path to the folder where all the components of the model are stored
	 * @param modelSource
	 * 	String path to the actual model file. In Pytorch is the path to a .pt file
	 * 	and for Tf it is the same as the modelFolder
     * @throws Exception if there is any problem loading the model, and the model
     * cannot be loaded
     */
    public void loadModel(String modelFolder, String modelSource) throws LoadModelException;
    
    /**
     * Closes the model loaded on the class on a particular ClassLoader
     */
    public void closeModel();
}
