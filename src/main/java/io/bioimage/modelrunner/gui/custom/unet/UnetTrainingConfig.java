/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.gui.custom.unet;

import java.io.File;

public final class UnetTrainingConfig {

    private final String modelName;
    private final String datasetPath;
    private final int epochs;
    private final boolean fineTune;
    private final String baseModelPath;
    private final String scratchArchitecture;
    private final String modelsDir;
    private final String outputModelDir;
    private final String device;

    /**
     * Creates a new UnetTrainingConfig instance.
     *
     * @param modelName the model name.
     * @param datasetPath the dataset path.
     * @param epochs the epochs.
     * @param fineTune the fineTune parameter.
     * @param baseModelPath the base model path.
     * @param scratchArchitecture the scratch architecture.
     * @param modelsDir the models directory.
     * @param outputModelDir the output model directory.
     * @param device the device.
     */
    public UnetTrainingConfig(String modelName, String datasetPath, int epochs,
            boolean fineTune, String baseModelPath, String scratchArchitecture,
            String modelsDir, String outputModelDir, String device) {
        this.modelName = modelName;
        this.datasetPath = datasetPath;
        this.epochs = epochs;
        this.fineTune = fineTune;
        this.baseModelPath = baseModelPath;
        this.scratchArchitecture = scratchArchitecture;
        this.modelsDir = modelsDir;
        this.outputModelDir = outputModelDir;
        this.device = normalizeDevice(device);
    }

    /**
     * Returns the model name.
     *
     * @return the model name.
     */
    public String getModelName() {
        return modelName;
    }

    /**
     * Returns the dataset path.
     *
     * @return the dataset path.
     */
    public String getDatasetPath() {
        return datasetPath;
    }

    /**
     * Returns the epochs.
     *
     * @return the epochs.
     */
    public int getEpochs() {
        return epochs;
    }

    /**
     * Returns whether fine tune.
     *
     * @return true if fine tune.
     */
    public boolean isFineTune() {
        return fineTune;
    }

    /**
     * Returns the base model path.
     *
     * @return the base model path.
     */
    public String getBaseModelPath() {
        return baseModelPath;
    }

    /**
     * Returns the scratch architecture.
     *
     * @return the scratch architecture.
     */
    public String getScratchArchitecture() {
        return scratchArchitecture;
    }

    /**
     * Returns the models directory.
     *
     * @return the models directory.
     */
    public String getModelsDir() {
        return modelsDir;
    }

    /**
     * Returns the output model directory.
     *
     * @return the output model directory.
     */
    public String getOutputModelDir() {
        return outputModelDir;
    }

    /**
     * Returns the model checkpoint path.
     *
     * @return the model checkpoint path.
     */
    public String getOutputModelPath() {
        return new File(outputModelDir, "model.pt").getAbsolutePath();
    }

    /**
     * Returns the device.
     *
     * @return the device.
     */
    public String getDevice() {
        return device;
    }

    /**
     * Creates a config from UI values.
     *
     * @param modelName the model name.
     * @param datasetPath the dataset path.
     * @param epochs the epochs.
     * @param fineTune the fine tune.
     * @param baseModelPath the base model path.
     * @param scratchArchitecture the scratch architecture.
     * @param modelsDir the models directory.
     * @param device the device.
     * @return the created config.
     */
    public static UnetTrainingConfig fromUi(String modelName, String datasetPath, int epochs,
            boolean fineTune, String baseModelPath, String scratchArchitecture, String modelsDir, String device) {
        String normalizedName = normalizeModelName(modelName);
        File unetDir = modelsDir == null
                ? new File(UnetModelRegistry.UNET_MODELS_SUBDIR)
                : new File(modelsDir, UnetModelRegistry.UNET_MODELS_SUBDIR);
        File outputDir = new File(unetDir, normalizedName);
        return new UnetTrainingConfig(normalizedName, datasetPath, epochs, fineTune,
                fineTune ? baseModelPath : null, fineTune ? null : scratchArchitecture,
                modelsDir, outputDir.getAbsolutePath(), device);
    }

    private static String normalizeModelName(String modelName) {
        if (modelName == null) {
            return "";
        }
        String name = modelName.trim();
        name = UnetModelRegistry.removeWeightsExtension(name);
        return name.trim();
    }

    private static String normalizeDevice(String device) {
        if (device == null) {
            return "cpu";
        }
        String normalized = device.trim().toLowerCase();
        return "cuda".equals(normalized) || "mps".equals(normalized) ? normalized : "cpu";
    }
}
