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
    private final String outputWeightsPath;
    private final int previewEpochPeriod;

    public YoloTrainingConfig(String modelName, String datasetYamlPath, int epochs, int imageSize,
            boolean fineTune, String baseModelPath, String outputWeightsPath, int previewEpochPeriod) {
        this.modelName = modelName;
        this.datasetYamlPath = datasetYamlPath;
        this.epochs = epochs;
        this.imageSize = imageSize;
        this.fineTune = fineTune;
        this.baseModelPath = baseModelPath;
        this.outputWeightsPath = outputWeightsPath;
        this.previewEpochPeriod = previewEpochPeriod;
    }

    public String getModelName() {
        return modelName;
    }

    public String getDatasetYamlPath() {
        return datasetYamlPath;
    }

    public int getEpochs() {
        return epochs;
    }

    public int getImageSize() {
        return imageSize;
    }

    public boolean isFineTune() {
        return fineTune;
    }

    public String getBaseModelPath() {
        return baseModelPath;
    }

    public String getOutputWeightsPath() {
        return outputWeightsPath;
    }

    public int getPreviewEpochPeriod() {
        return previewEpochPeriod;
    }

    public static YoloTrainingConfig fromUi(String modelName, String datasetPath, int epochs,
            boolean fineTune, String baseModelPath, String modelsDir) {
        String normalizedName = normalizeModelName(modelName);
        File yoloDir = modelsDir == null
                ? new File(YoloModelRegistry.YOLO_MODELS_SUBDIR)
                : new File(modelsDir, YoloModelRegistry.YOLO_MODELS_SUBDIR);
        File output = new File(yoloDir, normalizedName + YoloModelRegistry.YOLO_WEIGHTS_EXTENSION);
        return new YoloTrainingConfig(normalizedName, datasetPath, epochs, DEFAULT_IMAGE_SIZE,
                fineTune, fineTune ? baseModelPath : null, output.getAbsolutePath(), DEFAULT_PREVIEW_EPOCH_PERIOD);
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
}
