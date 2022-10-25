package org.bioimageanalysis.icy.deeplearning.transformations;


import java.util.stream.IntStream;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

public class ScaleRangeTransformation extends DefaultImageTransformation {
    public static final String name = "scale_range";
    /**
     * Compulsory parameter. Maximum percentile for the image
     */
    private Number maxPercentile;
    /**
     * Compulsory parameter. Minimum percentile
     */
    private Number minPercentile;
    /**
     * Compulsory, although it can be replaced by the INDArray input
     */
    private Tensor inputTensor;
    /**
     * Optional. Axes along which the transformation is calculated
     */
    private String axes;
    /**
     * Compulsory. Whether the method is applied per sample or per dataset
     */
    private String mode;
    /**
     * TODO find out which is this for
     * Optional. REference tensor
     */
    private String referenceTensor;

    /**
     * Constructor for the transformation using a Tensor for the processing
     *
     * @param input input in the form of Tensor
     */
    public ScaleRangeTransformation(Tensor input) {
        inputTensor = input;
    }

    /**
     * Set the maximum percentile
     *
     * @param maxPercentile maximum percentile
     */
    public void setMaxPercentile(Number maxPercentile) {
        this.maxPercentile = maxPercentile;
    }

    public void setMinPercentile(Number minPercentile) {
        this.minPercentile = minPercentile;
    }

    public void setAxes(String axes) {
        this.axes = axes;
    }

    public void setMode(String mode) {
        this.mode = mode;
    }

    public void setReferenceTensor(String referenceTensor) {
        this.referenceTensor = referenceTensor;
    }

    @Override
    public String getName() {
        return name;
    }

    private int[] getAxesForPercentileCalc() {
        String axesOrder = inputTensor.getAxesOrderString();
        if (axes == null)
            axes = axesOrder;
        return IntStream.range(0, axesOrder.length())
            .filter(i -> axes.indexOf(axesOrder.split("")[i]) != -1).toArray();
    }
    /**
     private int[] getSqueezedShape(int[] percentileAxes) {
     long[] squeezedShape = inputTensor.getData().shape();
     int[] shape = IntStream.range(0, squeezedShape.length).map(i -> 0 + (int) squeezedShape[i]).toArray();
     for (int i = 0; i < percentileAxes.length; i ++) {
     shape[percentileAxes[i]] = 1;
     }
     return shape;
     }

     private float getFloatVal(Number val) {
     return val.floatValue();
     }

     // TODO solve memory leak in max min calculation by nd4j, either solve it
     // with another method to find the max and min or try with wokspaces
     private INDArray getPercentileMat(int[] percentileAxes, Number perc) {
     INDArray array = inputTensor.getData();
     getMax(array, percentileAxes);
     INDArray maxP = array.max(percentileAxes);
     INDArray minP = array.min(percentileAxes);
     double constant = getFloatVal(perc) / 100.0;
     INDArray mat = minP.add((maxP.sub(minP)).mul(constant));
     int[] squeezedShape = getSqueezedShape(percentileAxes);
     INDArray mat2 = mat.reshape(squeezedShape).dup();//.broadcast(shape);
     mat.close();
     minP.close();
     maxP.close();
     return mat2;
     }

     private INDArray getMax(INDArray arr, int[] interestAxes) {
     int[] maxAxes = new int[arr.shape().length - interestAxes.length];
     long[] maxArrSize = new long[arr.shape().length - interestAxes.length];
     int c = 0;
     for (int i = 0; i < arr.shape().length; i ++) {
     boolean present = false;
     for (int j = 0; j < interestAxes.length; j ++) {
     if (i == j) {
     present = true;
     break;
     }
     }
     if (!present) {
     maxAxes[c] = i;
     maxArrSize[c ++] = arr.shape()[i];
     }
     }

     INDArray result = Nd4j.zeros(maxArrSize);
     return result;
     }

     private void checkCompulsoryArgs() {
     if (minPercentile == null || maxPercentile == null) {
     throw new IllegalArgumentException("Error defining the processing '"
     + name + "'. It should at least be provided with the "
     + "arguments 'min_percentile' and 'max_percetile' in the"
     + " yaml file.");
     }
     }

     public Tensor apply() {
     checkCompulsoryArgs();
     inputTensor.createCopyOfTensorInWantedDataType(DataType.FLOAT);
     // Get memory manager to remove arrays created from off-heap memory
     int[] percentileAxes = getAxesForPercentileCalc();

     MemoryManager mm = Nd4j.getMemoryManager();

     INDArray minPercMat = getPercentileMat(percentileAxes, minPercentile);
     INDArray maxPercMat = getPercentileMat(percentileAxes, maxPercentile);

     INDArray arr = inputTensor.getData();
     arr.sub(minPercMat).div(maxPercMat.sub(minPercMat), arr);

     minPercMat.data().flush();
     minPercMat.data().destroy();
     minPercMat = null;
     maxPercMat.data().flush();
     maxPercMat.data().destroy();
     maxPercMat = null;
     mm.invokeGc();
     return inputTensor;
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
     */
}
