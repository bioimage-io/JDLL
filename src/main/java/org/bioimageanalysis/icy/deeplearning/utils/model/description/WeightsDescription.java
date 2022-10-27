package org.bioimageanalysis.icy.deeplearning.utils.model.description;

public class WeightsDescription {
    private WeightDescription keras_hdf5;
    private WeightDescription tensorflow_saved_model_bundle;

    public WeightDescription getKeras_hdf5() {
        return keras_hdf5;
    }

    public void setKeras_hdf5(WeightDescription keras_hdf5) {
        this.keras_hdf5 = keras_hdf5;
    }

    public WeightDescription getTensorflow_saved_model_bundle() {
        return tensorflow_saved_model_bundle;
    }

    public void setTensorflow_saved_model_bundle(WeightDescription tensorflow_saved_model_bundle) {
        this.tensorflow_saved_model_bundle = tensorflow_saved_model_bundle;
    }
}
