package org.bioimageanalysis.icy.deeplearning.utils.model.description;

public class KwargsDescription {
    private String axes;
    private float max_percentile;
    private float min_percentile;
    private String mode;

    public String getAxes() {
        return axes;
    }

    public void setAxes(String axes) {
        this.axes = axes;
    }

    public float getMax_percentile() {
        return max_percentile;
    }

    public void setMax_percentile(float max_percentile) {
        this.max_percentile = max_percentile;
    }

    public float getMin_percentile() {
        return min_percentile;
    }

    public void setMin_percentile(float min_percentile) {
        this.min_percentile = min_percentile;
    }

    public String getMode() {
        return mode;
    }

    public void setMode(String mode) {
        this.mode = mode;
    }
}
