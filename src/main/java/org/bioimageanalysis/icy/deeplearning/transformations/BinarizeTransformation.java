package org.bioimageanalysis.icy.deeplearning.transformations;

import java.nio.FloatBuffer;

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
					+ "argument 'threshold' in the"
					+ " yaml file.");
		}
	}
	
	private float getFloatThreshold() {
		return threshold.floatValue();
	}
	
	public Tensor apply() {
		checkCompulsoryArgs();
		tensor.convertToDataType(DataType.FLOAT);
		FloatBuffer datab = tensor.getDataAsNDArray().data().asNioFloat();
		float thres = getFloatThreshold();
		for (int i = 0; i < tensor.getDataAsNDArray().length(); i ++) {
			float aa = datab.get(i) - thres;
			if (aa > 0 )
				datab.put(i, 1);
			else
				datab.put(i, 0);
		}
		return tensor;
	}
	
	public static void main(String[] args) throws InterruptedException {
		INDArray arr = Nd4j.arange(96000000);
		arr = arr.reshape(new int[] {2,3,4000,4000});
		Tensor tt = Tensor.build("example", "bcyx", arr);
		long t1 = System.currentTimeMillis();
		for (int i = 0; i < 1; i ++) {
			BinarizeTransformation preproc = new BinarizeTransformation(tt);
			preproc.setThreshold(960000000);
			preproc.apply();
		}
		System.out.println(System.currentTimeMillis() - t1);
		System.out.println("done");
	}
}
