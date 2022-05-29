package org.bioimageanalysis.icy.deeplearning.transformations;

import java.nio.FloatBuffer;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.nd4j.linalg.api.buffer.DataType;


public class ZeroMeanUnitVarianceTransformation extends DefaultImageTransformation {
	public static final String name = "zero_mean_unit_variance";
	private Number mean;
	private Number std;
	private Tensor input;
	private String axes;
	private String mode;

	public ZeroMeanUnitVarianceTransformation(Tensor input) {
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
	
	private float getFloatVal(Number val) {
		return val.floatValue();
	}
	
	/**
	 * 
	 * @param axes
	 * @param per_sample
	 * @return
	 */
	public Tensor apply() {
		FloatBuffer arr = input.getData().data().asNioFloat();
		float mean = 0;
		for (int i = 0; i < input.getData().length(); i ++)
			mean += arr.get();
		mean = mean / input.getData().length();
		float std = 0;
		for (int i = 0; i < input.getData().length(); i ++)
			std += ((arr.get(i) - mean) * (arr.get(i) - mean));
		
		std = std / input.getData().length();
		
		for (int i = 0; i < input.getData().length(); i ++) {
			arr.put(i, (arr.get(i) - mean) / std);
		}
		input.convertToDataType(DataType.FLOAT);
		return input;
	}
}
