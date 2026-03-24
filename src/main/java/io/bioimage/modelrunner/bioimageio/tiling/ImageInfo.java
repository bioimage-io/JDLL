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
package io.bioimage.modelrunner.bioimageio.tiling;

public class ImageInfo {
	
	private final String name;
	
	private final String axesOrder;
	
	private final long[] dims;
	
	/**
	 * Creates a new ImageInfo.
	 *
	 * @param tensorName the tensorName parameter.
	 * @param axesOrder the axesOrder parameter.
	 * @param dims the dims parameter.
	 */
	public ImageInfo(String tensorName, String axesOrder, long[] dims) {
		this.dims = dims;
		this.name = tensorName;
		this.axesOrder = axesOrder;
	}
	
	/**
	 * Gets axes order.
	 *
	 * @return the resulting string.
	 */
	public String getAxesOrder() {
		return this.axesOrder;
	}
	
	/**
	 * Gets tensor name.
	 *
	 * @return the resulting string.
	 */
	public String getTensorName() {
		return this.name;
	}
	
	/**
	 * Gets dimensions.
	 *
	 * @return the resulting array.
	 */
	public long[] getDimensions() {
		return this.dims;
	}

}
