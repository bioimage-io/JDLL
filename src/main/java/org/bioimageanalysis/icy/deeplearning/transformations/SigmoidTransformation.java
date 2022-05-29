package org.bioimageanalysis.icy.deeplearning.transformations;

import java.nio.FloatBuffer;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SigmoidTransformation extends DefaultImageTransformation {

	public static final String name = "sigmoid";
	
	private Tensor tensor;
	
	public SigmoidTransformation(Tensor tensor) {
		this.tensor = tensor;
	}

	@Override
	public String getName() {
		return name;
	}
	
	public Tensor apply() {
		// TODO Should it be directly converted to float 32?
		tensor.convertToDataType(DataType.FLOAT);
		FloatBuffer datab = tensor.getData().data().asNioFloat();
		for (int i = 0; i < tensor.getData().length(); i ++) {
			datab.put(i, (float) (1.0 / ( 1.0 + Math.exp(-datab.get(i)))));
		}
		return tensor;
	}
	
	public static void main(String[] args) throws InterruptedException {
		INDArray arr = Nd4j.arange(96000000);
		arr = arr.reshape(new int[] {2,3,4000,4000});
		Tensor tt = Tensor.build("example", "bcyx", arr);
		long t1 = System.currentTimeMillis();
		for (int i = 0; i < 1; i ++) {
			SigmoidTransformation preproc = new SigmoidTransformation(tt);
			preproc.apply();
		}
		System.out.println(System.currentTimeMillis() - t1);
		System.out.println("done");
	}
}
