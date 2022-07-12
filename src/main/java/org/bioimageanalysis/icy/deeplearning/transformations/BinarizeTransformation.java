package org.bioimageanalysis.icy.deeplearning.transformations;

import java.nio.FloatBuffer;

import org.bioimageanalysis.icy.deeplearning.tensor.RaiToArray;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.real.FloatType;

// TODO check efficiency
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
	
	public <T extends Type<T>> Tensor apply() {
		checkCompulsoryArgs();
		tensor = Tensor.createCopyOfTensorInWantedDataType(tensor, new FloatType());
		float[] arr = RaiToArray.floatArray(tensor.getData());
		FloatBuffer buff = FloatBuffer.wrap(arr);
		float thres = getFloatThreshold();
		for (int i = 0; i < arr.length; i ++) {
			float aa = buff.get(i) - thres;
			if (aa > 0 )
				buff.put(i, 1);
			else
				buff.put(i, 0);
		}
		tensor.setData(ArrayImgs.floats(arr, tensor.getData().dimensionsAsLongArray()));
		return tensor;
	}
	
	public static void main(String[] args) throws InterruptedException {
		Tensor tt = Tensor.build("example", "bcyx", null);
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
