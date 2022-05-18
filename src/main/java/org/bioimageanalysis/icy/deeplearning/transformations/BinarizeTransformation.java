package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.nd4j.linalg.api.buffer.DataType;

public class BinarizeTransformation extends DefaultImageTransformation {

	public static final String name = "binarize";
	private Number threshold;

	public void setThreshold(Number threshold) {
		this.threshold = threshold;
	}

	@Override
	public String getName() {
		return name;
	}
	
	private Tensor tensor;
	
	public BinarizeTransformation(Tensor tensor) {
		tensor.convertToDataType(DataType.FLOAT);
		float[] data = tensor.getDataAsNDArray().data().asFloat();
		float thres = (float) threshold;
		for (int i = 0; i < data.length; i ++) {
			data[i] = 2 * (data[i] - thres) / Math.abs((data[i] - thres)) - 1;
		}

		for (int i = 0; i < data.length; i ++) {
			float aa = data[i] - thres;
			if (aa > 0 )
				data[i] = 1;
			else
				data[i] = 0;
		}
		tensor.getDataAsNDArray().data().setData(data);
	}
	
	public Tensor apply() {
		return tensor;
	}
}
