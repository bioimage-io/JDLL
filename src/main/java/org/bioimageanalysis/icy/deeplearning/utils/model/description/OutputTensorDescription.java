package org.bioimageanalysis.icy.deeplearning.utils.model.description;

public class OutputTensorDescription {
    private String axes;
    private float[] data_range;
    private String data_type;
    private String name;
    private int[] shape;

    public String getAxes() {
        return axes;
    }

    public void setAxes(String axes) {
        this.axes = axes;
    }

    public float[] getData_range() {
        return data_range;
    }

    public void setData_range(float[] data_range) {
        this.data_range = data_range;
    }

    public String getData_type() {
        return data_type;
    }

    public void setData_type(String data_type) {
        this.data_type = data_type;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int[] getShape() {
        return shape;
    }

    public void setShape(int[] shape) {
        this.shape = shape;
    }
}
