package org.bioimageanalysis.icy.deeplearning.transformations;

import java.util.List;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

public class ScaleLinearTransformation extends DefaultImageTransformation {

	public static final String name = "scale_linear";
	private Number offset;
	private List<Number> offsetArr;
	private Number gain;
	private List<Number> gainArr;
	private String axes;
	
	private Tensor tensor;
	
	public ScaleLinearTransformation(Tensor tensor) {
		this.tensor = tensor;
	}


	public void setOffset(Number offset) {
		this.offset = offset;
	}


	public void setOffset(List<Number> offsetArr) {
		this.offsetArr = offsetArr;
	}

	public void setGain(Number gain) {
		this.gain = gain;
	}

	public void setGain(List<Number> gainArr) {
		this.gainArr = gainArr;
	}
	
	public void setAxes(String axes) {
		this.axes = axes;
	}

	@Override
	public String getName() {
		return name;
	}
	
	private float getFloatVal(Number val) {
		if (val instanceof Integer)
			return (float) (1.0 * (int) val);
		else if (val instanceof Float)
			return (float) val;
		else if (val instanceof Double)
			return (float) val;
		else if (val instanceof Long)
			return (float) (1.0 * (long) val);
		else 
			throw new IllegalArgumentException("Type '" + val.getClass().toString() + "' of the"
					+ " parameters for the processing '" + name + "' not supported.");
	}
	
	private void checkCompulsoryArgs() {
		if (gain == null || offset == null) {
			throw new IllegalArgumentException("Error defining the processing '"
					+ name + "'. It should at least be provided with the "
					+ "arguments 'gain' and 'offset' in the"
					+ " yaml file.");
		}
	}
	
	public Tensor apply() {
		checkCompulsoryArgs();
		return tensor;
	}
}
