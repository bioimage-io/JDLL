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

    public static final boolean DEFAULT_GPU = true;
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
    private final String labelColorMode;
    private final double validFraction;

    public StardistTrainingConfig(String modelName, String datasetPath, int epochs,
            boolean fineTune, String baseModelPath, String scratchArchitecture,
            String modelsDir, String outputModelDir, boolean gpu, String labelColorMode,
            double validFraction) {
        this.modelName = modelName;
        this.datasetPath = datasetPath;
        this.epochs = epochs;
        this.fineTune = fineTune;
        this.baseModelPath = baseModelPath;
        this.scratchArchitecture = scratchArchitecture;
        this.modelsDir = modelsDir;
        this.outputModelDir = outputModelDir;
        this.gpu = gpu;
        this.labelColorMode = labelColorMode;
        this.validFraction = validFraction;
    }

    public String getModelName() {
        return modelName;
    }

    public String getDatasetPath() {
        return datasetPath;
    }

    public int getEpochs() {
        return epochs;
    }

    public boolean isFineTune() {
        return fineTune;
    }

    public String getBaseModelPath() {
        return baseModelPath;
    }

    public String getScratchArchitecture() {
        return scratchArchitecture;
    }

    public String getModelsDir() {
        return modelsDir;
    }

    public String getOutputModelDir() {
        return outputModelDir;
    }

    public boolean isGpu() {
        return gpu;
    }

    public String getLabelColorMode() {
        return labelColorMode;
    }

    public double getValidFraction() {
        return validFraction;
    }

    public String getImageChannels() {
        String architecture = scratchArchitecture == null ? "" : scratchArchitecture.toLowerCase();
        return architecture.startsWith("color") ? "rgb" : "grayscale";
    }

    public static StardistTrainingConfig fromUi(String modelName, String datasetPath, int epochs,
            boolean fineTune, String baseModelPath, String scratchArchitecture, String modelsDir) {
        String normalizedName = normalizeModelName(modelName);
        File stardistDir = modelsDir == null
                ? new File(StardistModelRegistry.STARDIST_MODELS_SUBDIR)
                : new File(modelsDir, StardistModelRegistry.STARDIST_MODELS_SUBDIR);
        File output = new File(stardistDir, normalizedName);
        return new StardistTrainingConfig(normalizedName, datasetPath, epochs,
                fineTune, fineTune ? baseModelPath : null, fineTune ? null : scratchArchitecture,
                modelsDir, output.getAbsolutePath(), DEFAULT_GPU, DEFAULT_LABEL_COLOR_MODE,
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
}
