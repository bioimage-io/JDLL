package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

public class ClipTransformation extends DefaultImageTransformation {

	public static final String name = "clip";
	private Number min;
	private Number max;

	public Number getMin() {
		return min;
	}

	public void setMin(Number min) {
		this.min = min;
	}

	public Number getMax() {
		return max;
	}

	public void setMax(Number max) {
		this.max = max;
	}

	@Override
	public String getName() {
		return name;
	}
	
	private Tensor tensor;
	
	public ClipTransformation(Tensor tensor) {
		this.tensor = tensor;
	}
	
	public Tensor apply() {
		return tensor;
	}
}
