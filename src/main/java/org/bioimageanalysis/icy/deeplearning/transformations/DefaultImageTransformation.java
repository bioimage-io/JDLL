package org.bioimageanalysis.icy.deeplearning.transformations;

public abstract class DefaultImageTransformation implements ImageTransformation {

    private Mode mode;

    @Override
    public Mode getMode() {
        return mode;
    }

    @Override
    public void setMode(Mode mode) {
        this.mode = mode;
    }
}
