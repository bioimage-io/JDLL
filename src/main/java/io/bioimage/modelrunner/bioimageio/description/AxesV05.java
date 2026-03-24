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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public class AxesV05 implements Axes {
	
	private final List<Axis> axesList;
	
	private final String axesOrder;
	
	private final double[] scaleArr;
	
	private final int[] minArr;
	
	private final int[] stepArr;
	
	private int[] haloArr;
	
	/**
	 * Creates a new AxesV05.
	 *
	 * @param axesList the axesList parameter.
	 */
	protected AxesV05(List<Object> axesList) {
		List<Axis> axesListInit = new ArrayList<Axis>();
		String order = "";
		int[] minArr = new int[axesList.size()];
		int[] stepArr = new int[axesList.size()];
		double[] scaleArr = new double[axesList.size()];
		int c = 0;
		for (Object axisObject : axesList) {
			if (!(axisObject instanceof Map))
				throw new IllegalArgumentException("The input argument should be a list of maps. "
						+ "Go to the Bioimage.io specs documentation for more info.");
			Axis axis = new AxisV05((Map<String, Object>) axisObject);
			axesListInit.add(axis);
			order += axis.getAxis();
			minArr[c] = axis.getMin();
			stepArr[c] = axis.getStep();
			scaleArr[c ++] = axis.getScale();
		}
		this.axesList = axesListInit;
		this.axesOrder = order;
		this.scaleArr = scaleArr;
		this.minArr = minArr;
		this.stepArr = stepArr; 
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
	 * @return the axesList
	 */
	public List<Axis> getAxesList() {
		return axesList;
	}

	/**
	 * Gets min tile size arr.
	 *
	 * @return the resulting array.
	 */
	public int[] getMinTileSizeArr() {
		return this.minArr;
	}

	/**
	 * Gets tile step arr.
	 *
	 * @return the resulting array.
	 */
	public int[] getTileStepArr() {
		return this.stepArr;
	}

	/**
	 * Gets halo arr.
	 *
	 * @return the resulting array.
	 */
	public int[] getHaloArr() {
		haloArr = new int[this.axesList.size()];
		for (int i = 0; i < this.axesList.size(); i ++)
			haloArr[i] = this.axesList.get(i).getHalo();
		return this.haloArr;
	}

	/**
	 * Gets tile scale arr.
	 *
	 * @return the resulting array.
	 */
	public double[] getTileScaleArr() {
		return this.scaleArr;
	}
	
	/**
	 * Gets axis.
	 *
	 * @param abreviation the abreviation parameter.
	 * @return the resulting value.
	 */
	public Axis getAxis(String abreviation) {
		return axesList.stream().filter(ax -> ax.getAxis().equals(abreviation)).findFirst().orElse(null);
	}

}
