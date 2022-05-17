/*-
 * #%L
 * Java implementation of the bioimage.io model specification.
 * %%
 * Copyright (C) 2020 - 2021 Center for Systems Biology Dresden
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
package org.bioimageanalysis.icy.deeplearning.transformations;

import java.nio.Buffer;
import java.nio.FloatBuffer;

import org.nd4j.linalg.api.ndarray.INDArray;


public class ZeroMeanUnitVarianceTransformation extends DefaultImageTransformation {
	public static final String name = "zero_mean_unit_variance";
	private Number mean;
	private Number std;
	private INDArray input;
	private String axes;
	private String mode;

	public ZeroMeanUnitVarianceTransformation(INDArray input) {
		this.input = input;
	}

	public ZeroMeanUnitVarianceTransformation() {
	}
	
	public Number getMean() {
		return mean;
	}

	public void setMean(Number mean) {
		this.mean = mean;
	}

	public Number getStd() {
		return std;
	}

	public void setStd(Number std) {
		this.std = std;
	}
	
	public void setAxes(String axes){
		this.axes = axes;
	}
	
	public void setMode(String mode) {
		this.mode = mode;
	}

	@Override
	public String getName() {
		return name;
	}
	
	/**
	 * 
	 * @param axes
	 * @param per_sample
	 * @return
	 */
	public INDArray apply() {
		float[] arr = input.data().asFloat();
		float mean = 0;
		for (float i : arr)
			mean += i;
		mean = mean / (float) arr.length;
		float std = 0;
		for (float i : arr) {
			std += ((i - mean) * (i - mean));
		}
		std = std / (float) arr.length;
		
		for (int i = 0; i < arr.length; i ++) {
			arr[i] = (arr[i] - mean) / std;
		}
		input.data().setData(arr);
		return input;
	}
}
