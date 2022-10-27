package org.bioimageanalysis.icy.deeplearning.utils.model.description;

public class PreprocessingDescription {
    private KwargsDescription kwargs;
    private String name;

    public KwargsDescription getKwargs() {
        return kwargs;
    }

    public void setKwargs(KwargsDescription kwargs) {
        this.kwargs = kwargs;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
