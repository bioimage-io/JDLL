package org.bioimageanalysis.icy.deeplearning.utils;

public class InputTensorDescription {
    private String axes;
    private float[] daraRange;
    private String dataType;
    private String name;
    private PreprocessingDescription preprocessing;
    private int[] shape;

    public String getAxes() {
        return axes;
    }

    public float[] getDaraRange() {
        return daraRange;
    }

    public String getDataType() {
        return dataType;
    }

    public String getName() {
        return name;
    }

    public PreprocessingDescription getPreprocessing() {
        return preprocessing;
    }

    public int[] getShape() {
        return shape;
    }
}
