/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * #L%
 */
package io.bioimage.modelrunner.gui.custom.crossgoose;

import java.io.File;
import java.util.Locale;

import io.bioimage.modelrunner.gui.custom.training.TrainingModelPaths;
import io.bioimage.modelrunner.gui.custom.unet.DenseSegmentationTrainingConfig;

/** Values selected in the Cross-GOOSE training interface. */
public final class CrossGooseTrainingConfig implements DenseSegmentationTrainingConfig {

    private final String modelName;
    private final String datasetPath;
    private final int epochs;
    private final boolean fineTune;
    private final String baseModelPath;
    private final String scratchArchitecture;
    private final String modelsDir;
    private final String outputModelDir;
    private final String device;

    private CrossGooseTrainingConfig(String modelName, String datasetPath, int epochs,
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

    public static CrossGooseTrainingConfig fromUi(String modelName, String datasetPath,
            int epochs, boolean fineTune, String baseModelPath, String scratchArchitecture,
            String modelsDir, String device) {
        String normalizedName = CrossGooseModelRegistry.removeCheckpointExtension(
                modelName == null ? "" : modelName.trim()).trim();
        File familyRoot = modelsDir == null ? new File(CrossGooseModelRegistry.MODELS_SUBDIR)
                : new File(modelsDir, CrossGooseModelRegistry.MODELS_SUBDIR);
        File output = TrainingModelPaths.uniqueModelDir(familyRoot, normalizedName, ".ckpt");
        return new CrossGooseTrainingConfig(output.getName(), datasetPath, epochs, fineTune,
                fineTune ? baseModelPath : null,
                fineTune ? null : CrossGooseModelRegistry.DEFAULT_ARCHITECTURE,
                modelsDir, output.getAbsolutePath(), device);
    }

    @Override public String getModelName() { return modelName; }
    @Override public String getDatasetPath() { return datasetPath; }
    @Override public int getEpochs() { return epochs; }
    @Override public boolean isFineTune() { return fineTune; }
    @Override public String getBaseModelPath() { return baseModelPath; }
    @Override public String getScratchArchitecture() { return scratchArchitecture; }
    @Override public String getModelsDir() { return modelsDir; }
    @Override public String getOutputModelDir() { return outputModelDir; }
    @Override public String getDevice() { return device; }

    private static String normalizeDevice(String value) {
        String device = value == null ? "cpu" : value.trim().toLowerCase(Locale.ROOT);
        return "cuda".equals(device) || "mps".equals(device) ? device : "cpu";
    }
}
