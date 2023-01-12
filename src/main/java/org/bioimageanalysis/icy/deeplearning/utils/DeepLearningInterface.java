package org.bioimageanalysis.icy.deeplearning.utils;

import java.util.List;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import org.bioimageanalysis.icy.deeplearning.exceptions.LoadModelException;
import org.bioimageanalysis.icy.deeplearning.exceptions.RunModelException;

public interface DeepLearningInterface
{

	/**
	 * Method that the interface implements to make inference. In the class that
	 * implements the interface, the code to run the model on the tensors should
	 * go here.
	 * 
	 * @param inputTensors
	 *            list containing the input tensors
	 * @param outputTensors
	 *            list containing only the information about output tensors
	 * @return 
	 * @return output tensors produced by the model
	 * @throws RunModelException
	 *             if there is an error in the execution of the model
	 */
	/*
	 * TODO
	 * TODO
	 * TODO should it be like the commented option? ask jean-yves
	 */
	//public void run( List< Tensor > inputTensors, List< Tensor > outputTensors ) throws RunModelException;
	public void run( List< Tensor  < ? > > inputTensors, List< Tensor  < ? > > outputTensors ) throws RunModelException;

	/**
	 * Load the model with the corresponding engine on the particular
	 * independent ClassLoader. This is done to be able to load the model only
	 * one time and use it several times.
	 * 
	 * @param modelFolder
	 *            String path to the folder where all the components of the
	 *            model are stored
	 * @param modelSource
	 *            String path to the actual model file. In Pytorch is the path
	 *            to a .pt file and for Tf it is the same as the modelFolder
	 * @throws LoadModelException
	 *             if there is any problem loading the model, and the model
	 *             cannot be loaded
	 */
	public void loadModel( String modelFolder, String modelSource ) throws LoadModelException;

	/**
	 * Closes the model loaded on the class on a particular ClassLoader
	 */
	public void closeModel();
}
