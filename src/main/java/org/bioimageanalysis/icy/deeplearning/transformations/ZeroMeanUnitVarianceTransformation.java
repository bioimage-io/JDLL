package org.bioimageanalysis.icy.deeplearning.transformations;

import java.nio.FloatBuffer;

import org.bioimageanalysis.icy.deeplearning.tensor.RaiArrayUtils;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.real.FloatType;


public class ZeroMeanUnitVarianceTransformation extends DefaultImageTransformation {
    public static final String name = "zero_mean_unit_variance";
    private Number mean;
    private Number std;
    private Tensor tensor;
    private String axes;
    private String mode;

    public ZeroMeanUnitVarianceTransformation(Tensor tensor) {
        this.tensor = tensor;
    }

    public ZeroMeanUnitVarianceTransformation() {
    }

    public Number getMean() {
        return mean;
    }

    public void setMean(Number mean) {
        this.mean = mean;
    }

    public Number getStd() {
        return std;
    }

    public void setStd(Number std) {
        this.std = std;
    }

    public void setAxes(String axes) {
        this.axes = axes;
    }

    public void setMode(String mode) {
        this.mode = mode;
    }

    @Override
    public String getName() {
        return name;
    }

    private float getFloatVal(Number val) {
        return val.floatValue();
    }

    /**
     * @param axes
     * @param per_sample
     * @return
     */
    public Tensor apply() {
        tensor = Tensor.createCopyOfTensorInWantedDataType(tensor, new FloatType());
        float[] arr = RaiArrayUtils.floatArray(tensor.getData());
        FloatBuffer datab = FloatBuffer.wrap(arr);
        float mean = 0;
        for (int i = 0; i < arr.length; i++)
            mean += datab.get();
        mean = mean / arr.length;
        float std = 0;
        for (int i = 0; i < arr.length; i++)
            std += ((datab.get(i) - mean) * (datab.get(i) - mean));

        std = std / arr.length;

        for (int i = 0; i < arr.length; i++) {
            datab.put(i, (datab.get(i) - mean) / std);
        }
        long[] tensorShape = tensor.getData().dimensionsAsLongArray();
        tensor.setData(null);
        tensor.setData(ArrayImgs.floats(arr, tensorShape));
        return tensor;
    }
}
