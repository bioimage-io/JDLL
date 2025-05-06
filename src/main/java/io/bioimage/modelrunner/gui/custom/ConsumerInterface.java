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
package io.bioimage.modelrunner.gui.custom;

import java.util.List;

import javax.swing.JComponent;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * @author Carlos Garcia
 */
public abstract class ConsumerInterface {
	
	protected List<String> varNames;
	
	protected List<JComponent> componentsGui;
	
	public abstract String getModelsDir();
	
	public abstract void setComponents(List<JComponent> components);
	
	public abstract void setVarNames(List<String> componentNames);
	
	public abstract Object getFocusedImage();
	
	public abstract String getFocusedImageName();
	
	public abstract Integer getFocusedImageChannels();
	
	public abstract Integer getFocusedImageSlices();
	
	public abstract Integer getFocusedImageFrames();
	
	public abstract Integer getFocusedImageWidth();
	
	public abstract Integer getFocusedImageHeight();

	public abstract < T extends RealType< T > & NativeType< T > > RandomAccessibleInterval<T> getFocusedImageAsRai();
	
	public abstract < T extends RealType< T > & NativeType< T > > 
	void display(RandomAccessibleInterval<T> rai, String axes, String name);
	
	public void setVariableNames(List<String> varNames) {
		this.varNames = varNames;
	}
}
