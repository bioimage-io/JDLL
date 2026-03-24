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
package io.bioimage.modelrunner.gui.adapter;


import java.awt.Color;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Interface used by the consumer to adapt deepImageJ to the consumer specs
 */
public interface GuiAdapter {
	
	/**
	 * 
	 * @return the title of the software, it will appear in the header
	 */
	public String getSoftwareName();
	
	/**
	 * 
	 * @return in the header, subtitle of the title that should be a short sentence
	 */
	public String getSoftwareDescription();
	
	/**
	 * 
	 * @return the color we want the title
	 */
	public Color getTitleColor();
	
	/**
	 * 
	 * @return the color we want the subtitle
	 */
	public Color getSubtitleColor();
	
	/**
	 * 
	 * @return the background colour of the header
	 */
	public Color getHeaderColor();
	
	/**
	 * 
	 * @return the path in the resources folder to the icon that is going to be used
	 */
	public String getIconPath();
	
	/**
	 * 
	 * @return directory where the models are saved
	 */
	public String getModelsDir();
	
	/**
	 * 
	 * @return directory where engines are saved
	 */
	public String getEnginesDir();
	
	/**
	 * Create the {@link RunnerAdapter} used to run a model
	 * @param descriptor
	 * 	{@link ModelDescriptor} containing information about the model
	 * @return and object that can be used to run a model
	 * @throws IOException if there is any error related to engine installation for Python models
	 * @throws LoadEngineException if there is any error loading an engine or model
	 */
	public RunnerAdapter createRunner(ModelDescriptor descriptor) throws IOException, LoadEngineException;
	
	/**
	 * Create the {@link RunnerAdapter} used to run a model
	 * @param descriptor
	 * 	{@link ModelDescriptor} containing information about the model
	 * @param enginesPath
	 * 	path to the Java engines installation folder
	 * @return and object that can be used to run a model
	 * @throws IOException if there is any error related to engine installation for Python models
	 * @throws LoadEngineException if there is any error loading an engine or model
	 */
	public RunnerAdapter createRunner(ModelDescriptor descriptor, String enginesPath) throws IOException, LoadEngineException;
	
	/**
	 * This method should contain the code to convert a {@link RandomAccessibleInterval} into the consumer software
	 * image object and display it
	 *
	 * @param rai the rai parameter.
	 * @param axesOrder the axesOrder parameter.
	 * @param imTitle the imTitle parameter.
	 */
	public <T extends RealType<T> & NativeType<T>> void displayRai(RandomAccessibleInterval<T> rai, String axesOrder, String imTitle);

	/**
	 * Get the input tensors as described in the {@link ModelDescriptor}
	 *
	 * @param descriptor the descriptor parameter.
	 * @return the resulting list.
	 */
	public <T extends RealType<T> & NativeType<T>> List<Tensor<T>> getInputTensors(ModelDescriptor descriptor);

	/**
	 * 
	 * @return a list of the images that are going to be used as inputs to the model in the order 
	 * specified in the {@link ModelDescriptor} specs of the Bioimage.io model.
	 */
	public List<String> getInputImageNames();
	
	/**
	 * Convert a map containing the consumer software image objects into the tensors required for a model
	 *
	 * @param inputs the inputs parameter.
	 * @param descriptor the descriptor parameter.
	 * @return the resulting list.
	 */
	public <T extends RealType<T> & NativeType<T>> List<Tensor<T>> convertToInputTensors(Map<String, Object> inputs, ModelDescriptor descriptor);

	/**
	 * Notify to the consumer software which model has been run
	 * @param modelAbsPath
	 * 	the absolute path of the model that has been run
	 */
	public void notifyModelUsed(String modelAbsPath);
}
