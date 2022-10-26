package org.bioimageanalysis.icy.deeplearning.utils;

public class ModelDescription {

    private String description;
    private String documentation;
    private String formatVersion;
    InputTensorDescription inputs;
    OutputTensorDescription outputs;

    public String getDocumentation() {
        return documentation;
    }

    public String getDescription() {
        return description;
    }

    public String getFormatVersion() {
        return formatVersion;
    }

    public InputTensorDescription getInputs() {
        return inputs;
    }

    public OutputTensorDescription getOutputs() {
        return outputs;
    }
}
