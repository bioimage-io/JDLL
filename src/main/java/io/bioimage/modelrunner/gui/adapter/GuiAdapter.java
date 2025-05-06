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

public interface GuiAdapter {
	
	public String getSoftwareName();
	
	public String getSoftwareDescription();
	
	public Color getTitleColor();
	
	public Color getSubtitleColor();
	
	public Color getHeaderColor();
	
	public String getIconPath();
	
	public String getModelsDir();
	
	public String getEnginesDir();
	
	public RunnerAdapter createRunner(ModelDescriptor descriptor) throws IOException, LoadEngineException;
	
	public RunnerAdapter createRunner(ModelDescriptor descriptor, String enginesPath) throws IOException, LoadEngineException;
	
	public <T extends RealType<T> & NativeType<T>> void displayRai(RandomAccessibleInterval<T> rai, String axesOrder, String imTitle);

	public <T extends RealType<T> & NativeType<T>> List<Tensor<T>> getInputTensors(ModelDescriptor descriptor);

	public List<String> getInputImageNames();
	
	public <T extends RealType<T> & NativeType<T>> List<Tensor<T>> convertToInputTensors(Map<String, Object> inputs, ModelDescriptor descriptor);
}
