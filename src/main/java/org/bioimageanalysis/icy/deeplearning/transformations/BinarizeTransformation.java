package org.bioimageanalysis.icy.deeplearning.transformations;

public class BinarizeTransformation extends DefaultImageTransformation {

	public static final String name = "binarize";
	private Number threshold;

	public Number getThreshold() {
		return threshold;
	}

	public void setThreshold(Number threshold) {
		this.threshold = threshold;
	}

	@Override
	public String getName() {
		return name;
	}
}
