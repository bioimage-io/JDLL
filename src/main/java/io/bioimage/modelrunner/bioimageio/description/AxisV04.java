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
package io.bioimage.modelrunner.bioimageio.description;

import java.util.List;

public class AxisV04 implements Axis {

	private String description = "";
	private final String abreviation;
	private boolean concat = false;
	private double scale = 1.0;
	private int min = 1;
	private int step = 1;
	protected int halo = 0;
	private double offset = 0;
	String referenceTensor;
	String referenceAxis;
	
	
	/**
	 * Creates a new AxisV04.
	 *
	 * @param abreviation the abreviation parameter.
	 * @param min the min parameter.
	 * @param step the step parameter.
	 * @param halo the halo parameter.
	 * @param offset the offset parameter.
	 * @param scale the scale parameter.
	 * @param ref the ref parameter.
	 */
	protected AxisV04(String abreviation, int min, int step, int halo, double offset, double scale, String ref) {
		this.abreviation = abreviation;
		this.halo = halo;
		this.min = min;
		this.offset = offset;
		this.scale = scale;
		this.step = step;
		this.referenceAxis = abreviation;
		this.referenceTensor = ref;
	}
	
	/**
	 * Gets axis.
	 *
	 * @return the resulting string.
	 */
	public String getAxis() {
		return this.abreviation;
	}
	
	/**
	 * Gets min.
	 *
	 * @return the resulting numeric value.
	 */
	public int getMin() {
		return this.min;
	}
	
	/**
	 * Gets step.
	 *
	 * @return the resulting numeric value.
	 */
	public int getStep() {
		return this.step;
	}
	
	/**
	 * Gets scale.
	 *
	 * @return the resulting numeric value.
	 */
	public double getScale() {
		return this.scale;
	}

	/**
	 * @return the channelNames
	 */
	public List<String> getChannelNames() {
		return null;
	}

	/**
	 * @return the description
	 */
	public String getDescription() {
		return description;
	}

	/**
	 * @return the concat
	 */
	public boolean isConcat() {
		return concat;
	}
	
	/**
	 * Gets halo.
	 *
	 * @return the resulting numeric value.
	 */
	public int getHalo() {
		return this.halo;
	}
	
	/**
	 * Gets offset.
	 *
	 * @return the resulting numeric value.
	 */
	public double getOffset() {
		return this.offset;
	}
	
	/**
	 * Gets reference tensor.
	 *
	 * @return the resulting string.
	 */
	public String getReferenceTensor() {
		return this.referenceTensor;
	}
	
	/**
	 * Gets reference axis.
	 *
	 * @return the resulting string.
	 */
	public String getReferenceAxis() {
		return this.referenceAxis;
	}

}
