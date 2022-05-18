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
	
	public Tensor apply() {
		checkCompulsoryArgs();
		tensor.convertToDataType(DataType.FLOAT);
		float[] data = tensor.getDataAsNDArray().data().asFloat();
		float thres = (float) threshold;
		long t1 = System.currentTimeMillis();
		for (int i = 0; i < data.length; i ++) {
			data[i] = 2 * (data[i] - thres) / Math.abs((data[i] - thres)) - 1;
		}
		long t2 = System.currentTimeMillis();
		for (int i = 0; i < data.length; i ++) {
			float aa = data[i] - thres;
			if (aa > 0 )
				data[i] = 1;
			else
				data[i] = 0;
		}
		long t3 = System.currentTimeMillis();
		long aa = t2 - t1;
		long bb = t3 - t2;
		tensor.getDataAsNDArray().data().setData(data);
		return tensor;
	}
	
	public static void main(String[] args) throws InterruptedException {
		INDArray arr = Nd4j.arange(96000000);
		arr = arr.reshape(new int[] {2,3,4000,4000});
		Tensor tt = Tensor.build("example", "bcyx", arr);
		ScaleRangeTransformation preproc = new ScaleRangeTransformation(tt);
		preproc.setMinPercentile(10);
		preproc.setMaxPercentile(90);
		preproc.setAxes("bc");
		preproc.apply();
		System.out.println();
	}
}
