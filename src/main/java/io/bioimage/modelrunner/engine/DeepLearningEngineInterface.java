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
package io.bioimage.modelrunner.engine;

import java.util.List;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;

public interface DeepLearningEngineInterface
{
	/**
	 * Simply run inference on the images provided. If the dimensions, number, data type or other
	 * characteristic of the tensor is not correct, an exception will be thrown.
	 *
	 * @param <T> the T type parameter.
	 * @param <R> the R type parameter.
	 * @param inputs the inputs to process.
	 * @return the resulting list.
	 * @throws RunModelException if model inference cannot be run.
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<RandomAccessibleInterval<R>> inference(List<RandomAccessibleInterval<T>> inputs) throws RunModelException;

	/**
	 * Method that the interface implements to make inference. In the class that
	 * implements the interface, the code to run the model on the tensors should
	 * go here.
	 *
	 * @param <T> the T type parameter.
	 * @param <R> the R type parameter.
	 * @param inputTensors the input tensors to process.
	 * @param outputTensors the output tensors to populate.
	 * @throws RunModelException if model inference cannot be run.
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	void run( List< Tensor  < T > > inputTensors, List< Tensor  < R > > outputTensors ) throws RunModelException;

	/**
	 * Load the model with the corresponding engine on the particular
	 * independent ClassLoader. This is done to be able to load the model only
	 * one time and use it several times.
	 *
	 * @param modelFolder the model folder.
	 * @param modelSource the model source.
	 * @throws LoadModelException if the model cannot be loaded.
	 */
	public void loadModel( String modelFolder, String modelSource ) throws LoadModelException;

	/**
	 * Closes the model loaded on the class on a particular ClassLoader
	 */
	public void closeModel();
}
