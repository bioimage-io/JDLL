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
package io.bioimage.modelrunner.engine;

import java.util.List;

import io.bioimage.modelrunner.tensor.Tensor;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;

public interface DeepLearningEngineInterface
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
