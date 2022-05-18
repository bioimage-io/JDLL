package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class BinarizeTransformation extends DefaultImageTransformation {

	public static final String name = "binarize";
	private Number threshold;
	
	private Tensor tensor;
	
	public BinarizeTransformation(Tensor tensor) {
		this.tensor = tensor;
	}

	public void setThreshold(Number threshold) {
		this.threshold = threshold;
	}

	@Override
	public String getName() {
		return name;
	}
	
	private void checkCompulsoryArgs() {
		if (threshold == null) {
			throw new IllegalArgumentException("Error defining the processing '"
					+ name + "'. It should at least be provided with the "
					+ "arguments 'threshold'"
					+ " yaml file.");
		}
	}
	
	private float getFloatThreshold() {
		if (threshold instanceof Integer)
			return (float) (1.0 * (int) threshold);
		else if (threshold instanceof Float)
			return (float) threshold;
		else if (threshold instanceof Double)
			return (float) threshold;
		else if (threshold instanceof Long)
			return (float) (1.0 * (long) threshold);
		else 
			throw new IllegalArgumentException("Type '" + threshold.getClass().toString() + "' of the"
					+ " threshold parameter for processing '"
					+ name + "' not supported");
	}
	
	public Tensor apply() {
		checkCompulsoryArgs();
		tensor.convertToDataType(DataType.FLOAT);
		float[] data = tensor.getDataAsNDArray().data().asFloat();
		float thres = getFloatThreshold();
		for (int i = 0; i < data.length; i ++) {
			float aa = data[i] - thres;
			if (aa > 0 )
				data[i] = 1;
			else
				data[i] = 0;
		}
		tensor.getDataAsNDArray().data().setData(data);
		data = new float[] {};
		System.gc();
		return tensor;
	}
	
	public static void main(String[] args) throws InterruptedException {
		INDArray arr = Nd4j.arange(96000000);
		arr = arr.reshape(new int[] {2,3,4000,4000});
		Tensor tt = Tensor.build("example", "bcyx", arr);
		BinarizeTransformation preproc = new BinarizeTransformation(tt);
		preproc.setThreshold(10);
		preproc.apply();
		System.out.println();
	}
}
