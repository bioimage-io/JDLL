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
