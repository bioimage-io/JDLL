/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the BioImage.io nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
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
