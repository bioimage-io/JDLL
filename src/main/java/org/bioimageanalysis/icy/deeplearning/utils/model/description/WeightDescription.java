package org.bioimageanalysis.icy.deeplearning.utils.model.description;

public class WeightDescription {
    private String sha256;
    private String source;
    private String tensorflow_version;

    public String getSha256() {
        return sha256;
    }

    public void setSha256(String sha256) {
        this.sha256 = sha256;
    }

    public String getSource() {
        return source;
    }

    public void setSource(String source) {
        this.source = source;
    }

    public String getTensorflow_version() {
        return tensorflow_version;
    }

    public void setTensorflow_version(String tensorflow_version) {
        this.tensorflow_version = tensorflow_version;
    }
}
