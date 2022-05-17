package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

public class ScaleLinearTransformation extends DefaultImageTransformation {

	public static final String name = "scale_linear";
	private Number offset;
	private Number gain;

	public Number getOffset() {
		return offset;
	}

	public void setOffset(Number offset) {
		this.offset = offset;
	}

	public Number getGain() {
		return gain;
	}

	public void setGain(Number gain) {
		this.gain = gain;
	}

	@Override
	public String getName() {
		return name;
	}
	
	private Tensor tensor;
	
	public ScaleLinearTransformation(Tensor tensor) {
		this.tensor = tensor;
	}
	
	public Tensor apply() {
		return tensor;
	}
}
