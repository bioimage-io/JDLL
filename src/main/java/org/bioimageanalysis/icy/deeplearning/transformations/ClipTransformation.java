package org.bioimageanalysis.icy.deeplearning.transformations;

import java.nio.FloatBuffer;

import org.bioimageanalysis.icy.deeplearning.tensor.RaiArrayUtils;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.real.FloatType;

public class ClipTransformation extends DefaultImageTransformation {

	public static final String name = "clip";
	private Number min;
	private Number max;
	
	private Tensor tensor;
	
	public ClipTransformation(Tensor tensor) {
		this.tensor = tensor;
	}

	public void setMin(Number min) {
		this.min = min;
	}

	public void setMax(Number max) {
		this.max = max;
	}

	@Override
	public String getName() {
		return name;
	}
	
	private void checkCompulsoryArgs() {
		if (max == null || min == null) {
			throw new IllegalArgumentException("Error defining the processing '"
					+ name + "'. It should at least be provided with the "
					+ "arguments 'min' and 'max' in the"
					+ " yaml file.");
		}
	}
	
	private float getFloatVal(Number val) {
		return val.floatValue();
	}
	
	private void checkArgsCompatible() {
		if (getFloatVal(min) > getFloatVal(max))
			throw new IllegalArgumentException("The argument 'min' for the processing routine "
					+ "'" + name + "' cannot be bigger than the other parameter 'max'.");
	}
	
	public Tensor apply() {
		checkCompulsoryArgs();
		checkArgsCompatible();
		tensor = Tensor.createCopyOfTensorInWantedDataType(tensor, new FloatType());
		float[] arr = RaiArrayUtils.floatArray(tensor.getData());
		FloatBuffer datab = FloatBuffer.wrap(arr);
		float minF = getFloatVal(min);
		float maxF = getFloatVal(max);
		for (int i = 0; i < arr.length; i ++) {
			if (datab.get(i) > maxF)
				datab.put(maxF);
			else if (datab.get(i) < minF)
				datab.put(minF);
		}
		tensor.setData(ArrayImgs.floats(arr, tensor.getData().dimensionsAsLongArray()));
		return tensor;
	}
	
	public static void main(String[] args) throws InterruptedException {
		Tensor tt = Tensor.build("example", "bcyx", null);
		long t1 = System.currentTimeMillis();
		for (int i = 0; i < 1; i ++) {
			ClipTransformation preproc = new ClipTransformation(tt);
			preproc.setMin(100);
			preproc.setMax(10000);
			preproc.apply();
		}
		System.out.println(System.currentTimeMillis() - t1);
		System.out.println("done");
	}
}
