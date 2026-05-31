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
package io.bioimage.modelrunner.gui.custom.stardist;

import java.io.File;

public final class StardistTrainingConfig {

    public static final String DEFAULT_LABEL_COLOR_MODE = "grayscale";
    public static final double DEFAULT_VALID_FRACTION = 0.15d;

    private final String modelName;
    private final String datasetPath;
    private final int epochs;
    private final boolean fineTune;
    private final String baseModelPath;
    private final String scratchArchitecture;
    private final String modelsDir;
    private final String outputModelDir;
    private final boolean gpu;
    private final String device;
    private final String labelColorMode;
    private final double validFraction;

    /**
     * Creates a new StardistTrainingConfig instance.
     *
     * @param modelName the model name.
     * @param datasetPath the dataset path.
     * @param epochs the epochs.
     * @param fineTune the fine tune.
     * @param baseModelPath the base model path.
     * @param scratchArchitecture the scratch architecture.
     * @param modelsDir the models directory.
     * @param outputModelDir the output model directory.
     * @param gpu whether to use GPU.
     * @param labelColorMode the label color mode.
     * @param validFraction the valid fraction.
     */
    public StardistTrainingConfig(String modelName, String datasetPath, int epochs,
            boolean fineTune, String baseModelPath, String scratchArchitecture,
            String modelsDir, String outputModelDir, boolean gpu, String labelColorMode,
            double validFraction) {
        this(modelName, datasetPath, epochs, fineTune, baseModelPath, scratchArchitecture, modelsDir,
                outputModelDir, gpu ? "cuda" : "cpu", labelColorMode, validFraction);
    }

    /**
     * Creates a new StardistTrainingConfig instance.
     *
     * @param modelName the model name.
     * @param datasetPath the dataset path.
     * @param epochs the epochs.
     * @param fineTune the fine tune.
     * @param baseModelPath the base model path.
     * @param scratchArchitecture the scratch architecture.
     * @param modelsDir the models directory.
     * @param outputModelDir the output model directory.
     * @param device the device.
     * @param labelColorMode the label color mode.
     * @param validFraction the valid fraction.
     */
    public StardistTrainingConfig(String modelName, String datasetPath, int epochs,
            boolean fineTune, String baseModelPath, String scratchArchitecture,
            String modelsDir, String outputModelDir, String device, String labelColorMode,
            double validFraction) {
        this.modelName = modelName;
        this.datasetPath = datasetPath;
        this.epochs = epochs;
        this.fineTune = fineTune;
        this.baseModelPath = baseModelPath;
        this.scratchArchitecture = scratchArchitecture;
        this.modelsDir = modelsDir;
        this.outputModelDir = outputModelDir;
        this.device = normalizeDevice(device);
        this.gpu = !"cpu".equals(this.device);
        this.labelColorMode = labelColorMode;
        this.validFraction = validFraction;
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
     * @return true if fine tune; false otherwise.
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
     * Returns whether GPU.
     *
     * @return true if GPU; false otherwise.
     */
    public boolean isGpu() {
        return gpu;
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
     * Returns the label color mode.
     *
     * @return the label color mode.
     */
    public String getLabelColorMode() {
        return labelColorMode;
    }

    /**
     * Returns the valid fraction.
     *
     * @return the valid fraction.
     */
    public double getValidFraction() {
        return validFraction;
    }

    /**
     * Returns the image channels.
     *
     * @return the image channels.
     */
    public String getImageChannels() {
        String architecture = scratchArchitecture == null ? "" : scratchArchitecture.toLowerCase();
        return architecture.startsWith("color") ? "rgb" : "grayscale";
    }

    /**
     * Creates a StardistTrainingConfig from the ui.
     *
     * @param modelName the model name.
     * @param datasetPath the dataset path.
     * @param epochs the epochs.
     * @param fineTune the fine tune.
     * @param baseModelPath the base model path.
     * @param scratchArchitecture the scratch architecture.
     * @param modelsDir the models directory.
     * @return the created stardist training config.
     */
    public static StardistTrainingConfig fromUi(String modelName, String datasetPath, int epochs,
            boolean fineTune, String baseModelPath, String scratchArchitecture, String modelsDir) {
        return fromUi(modelName, datasetPath, epochs, fineTune, baseModelPath, scratchArchitecture, modelsDir, "cpu");
    }

    /**
     * Creates a StardistTrainingConfig from the ui.
     *
     * @param modelName the model name.
     * @param datasetPath the dataset path.
     * @param epochs the epochs.
     * @param fineTune the fine tune.
     * @param baseModelPath the base model path.
     * @param scratchArchitecture the scratch architecture.
     * @param modelsDir the models directory.
     * @param device the device.
     * @return the created stardist training config.
     */
    public static StardistTrainingConfig fromUi(String modelName, String datasetPath, int epochs,
            boolean fineTune, String baseModelPath, String scratchArchitecture, String modelsDir, String device) {
        String normalizedName = normalizeModelName(modelName);
        File stardistDir = modelsDir == null
                ? new File(StardistModelRegistry.STARDIST_MODELS_SUBDIR)
                : new File(modelsDir, StardistModelRegistry.STARDIST_MODELS_SUBDIR);
        File output = new File(stardistDir, normalizedName);
        return new StardistTrainingConfig(normalizedName, datasetPath, epochs,
                fineTune, fineTune ? baseModelPath : null, fineTune ? null : scratchArchitecture,
                modelsDir, output.getAbsolutePath(), device, DEFAULT_LABEL_COLOR_MODE,
                DEFAULT_VALID_FRACTION);
    }

    private static String normalizeModelName(String modelName) {
        if (modelName == null) {
            return "";
        }
        String name = modelName.trim();
        if (name.toLowerCase().endsWith(StardistModelRegistry.STARDIST_KERAS_WEIGHTS_EXTENSION)) {
            name = name.substring(0, name.length() - StardistModelRegistry.STARDIST_KERAS_WEIGHTS_EXTENSION.length());
        }
        return name;
    }

    private static String normalizeDevice(String device) {
        if (device == null) {
            return "cpu";
        }
        String normalized = device.trim().toLowerCase();
        return "cuda".equals(normalized) || "mps".equals(normalized) ? normalized : "cpu";
    }
}
