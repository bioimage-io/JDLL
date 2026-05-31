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
package io.bioimage.modelrunner.gui.custom.yolo;

import java.io.File;

public final class YoloTrainingConfig {

    public static final int DEFAULT_IMAGE_SIZE = 640;
    public static final int DEFAULT_PREVIEW_EPOCH_PERIOD = 1;

    private final String modelName;
    private final String datasetYamlPath;
    private final int epochs;
    private final int imageSize;
    private final boolean fineTune;
    private final String baseModelPath;
    private final String scratchArchitecture;
    private final String modelsDir;
    private final String outputWeightsPath;
    private final int previewEpochPeriod;
    private final String device;

    /**
     * Creates a new YoloTrainingConfig instance.
     *
     * @param modelName the model name.
     * @param datasetYamlPath the dataset YAML path.
     * @param epochs the epochs.
     * @param imageSize the image size.
     * @param fineTune the fine tune.
     * @param baseModelPath the base model path.
     * @param scratchArchitecture the scratch architecture.
     * @param modelsDir the models directory.
     * @param outputWeightsPath the output weights path.
     * @param previewEpochPeriod the preview epoch period.
     */
    public YoloTrainingConfig(String modelName, String datasetYamlPath, int epochs, int imageSize,
            boolean fineTune, String baseModelPath, String scratchArchitecture,
            String modelsDir, String outputWeightsPath, int previewEpochPeriod) {
        this(modelName, datasetYamlPath, epochs, imageSize, fineTune, baseModelPath, scratchArchitecture,
                modelsDir, outputWeightsPath, previewEpochPeriod, "cpu");
    }

    /**
     * Creates a new YoloTrainingConfig instance.
     *
     * @param modelName the model name.
     * @param datasetYamlPath the dataset YAML path.
     * @param epochs the epochs.
     * @param imageSize the image size.
     * @param fineTune the fine tune.
     * @param baseModelPath the base model path.
     * @param scratchArchitecture the scratch architecture.
     * @param modelsDir the models directory.
     * @param outputWeightsPath the output weights path.
     * @param previewEpochPeriod the preview epoch period.
     * @param device the device.
     */
    public YoloTrainingConfig(String modelName, String datasetYamlPath, int epochs, int imageSize,
            boolean fineTune, String baseModelPath, String scratchArchitecture,
            String modelsDir, String outputWeightsPath, int previewEpochPeriod, String device) {
        this.modelName = modelName;
        this.datasetYamlPath = datasetYamlPath;
        this.epochs = epochs;
        this.imageSize = imageSize;
        this.fineTune = fineTune;
        this.baseModelPath = baseModelPath;
        this.scratchArchitecture = scratchArchitecture;
        this.modelsDir = modelsDir;
        this.outputWeightsPath = outputWeightsPath;
        this.previewEpochPeriod = previewEpochPeriod;
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
     * Returns the dataset YAML path.
     *
     * @return the dataset YAML path.
     */
    public String getDatasetYamlPath() {
        return datasetYamlPath;
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
     * Returns the image size.
     *
     * @return the image size.
     */
    public int getImageSize() {
        return imageSize;
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
     * Returns the output weights path.
     *
     * @return the output weights path.
     */
    public String getOutputWeightsPath() {
        return outputWeightsPath;
    }

    /**
     * Returns the preview epoch period.
     *
     * @return the preview epoch period.
     */
    public int getPreviewEpochPeriod() {
        return previewEpochPeriod;
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
     * Creates a YoloTrainingConfig from the ui.
     *
     * @param modelName the model name.
     * @param datasetPath the dataset path.
     * @param epochs the epochs.
     * @param fineTune the fine tune.
     * @param baseModelPath the base model path.
     * @param scratchArchitecture the scratch architecture.
     * @param modelsDir the models directory.
     * @return the created yolo training config.
     */
    public static YoloTrainingConfig fromUi(String modelName, String datasetPath, int epochs,
            boolean fineTune, String baseModelPath, String scratchArchitecture, String modelsDir) {
        return fromUi(modelName, datasetPath, epochs, fineTune, baseModelPath, scratchArchitecture, modelsDir, "cpu");
    }

    /**
     * Creates a YoloTrainingConfig from the ui.
     *
     * @param modelName the model name.
     * @param datasetPath the dataset path.
     * @param epochs the epochs.
     * @param fineTune the fine tune.
     * @param baseModelPath the base model path.
     * @param scratchArchitecture the scratch architecture.
     * @param modelsDir the models directory.
     * @param device the device.
     * @return the created yolo training config.
     */
    public static YoloTrainingConfig fromUi(String modelName, String datasetPath, int epochs,
            boolean fineTune, String baseModelPath, String scratchArchitecture, String modelsDir, String device) {
        String normalizedName = normalizeModelName(modelName);
        File yoloDir = modelsDir == null
                ? new File(YoloModelRegistry.YOLO_MODELS_SUBDIR)
                : new File(modelsDir, YoloModelRegistry.YOLO_MODELS_SUBDIR);
        File output = new File(new File(yoloDir, normalizedName),
                normalizedName + YoloModelRegistry.YOLO_WEIGHTS_EXTENSION);
        return new YoloTrainingConfig(normalizedName, datasetPath, epochs, DEFAULT_IMAGE_SIZE,
                fineTune, fineTune ? baseModelPath : null, fineTune ? null : scratchArchitecture,
                modelsDir, output.getAbsolutePath(), DEFAULT_PREVIEW_EPOCH_PERIOD, device);
    }

    private static String normalizeModelName(String modelName) {
        if (modelName == null) {
            return "";
        }
        String name = modelName.trim();
        if (name.toLowerCase().endsWith(YoloModelRegistry.YOLO_WEIGHTS_EXTENSION)) {
            name = name.substring(0, name.length() - YoloModelRegistry.YOLO_WEIGHTS_EXTENSION.length());
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
