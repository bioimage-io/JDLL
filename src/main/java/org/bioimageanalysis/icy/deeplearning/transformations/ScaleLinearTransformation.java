package org.bioimageanalysis.icy.deeplearning.transformations;

import java.util.List;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

// TODO add functionality of offset and Gain being arrays
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
		return val.floatValue();
	}
	
	// TODO
	private void checkCompulsoryArgs() {
		if (gain == null || offset == null) {
			throw new IllegalArgumentException("Error defining the processing '"
					+ name + "'. It should at least be provided with the "
					+ "arguments 'gain' and 'offset' in the"
					+ " yaml file.");
		}
		if ((gainArr == null & gain == null) || (offsetArr == null && offset == null)) {
			throw new IllegalArgumentException("Error defining the processing '"
					+ name + "'. It should at least be provided with the "
					+ "arguments 'gain' and 'offset' in the"
					+ " yaml file.");
		}
	}
	
	// TODO
	private void checkArgsCompatibility() {
		if (axes == null && gainArr != null && gainArr.size() != 1
				&& offsetArr != null && offsetArr.size() != 1) {
			throw new IllegalArgumentException("In order to specify the 'gain' and 'offset' parameters "
					+ "for several axes in the processing '" + name + "', please provide the 'axes' parameter too.");
		} //else if (axes != null && (axes))
	}
	
	/**
	public Tensor apply() {
		checkCompulsoryArgs();
		tensor.createCopyOfTensorInWantedDataType(DataType.FLOAT);
		INDArray arr = tensor.getData();
		arr.mul(gain, arr);
		arr.add(offset, arr);
		return tensor;
	}
	*/
}
