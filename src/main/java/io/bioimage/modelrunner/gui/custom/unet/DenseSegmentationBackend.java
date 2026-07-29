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
package io.bioimage.modelrunner.gui.custom.unet;

import java.io.File;
import java.util.LinkedHashMap;

/** Family-specific behavior behind the shared dense-segmentation interface. */
public interface DenseSegmentationBackend {

    String getDisplayName();
    String getMaskSuffix();
    String getModelFileDescription();
    String[] getModelFileExtensions();
    LinkedHashMap<String, String> buildModelEntries(String modelsDir);
    File findModelFile(File path);
    String modelLabel(File selected);
    void configureTrainingPanel(UnetTrainPanel panel, String modelsDir);
    void refreshTrainingPanel(UnetTrainPanel panel, String modelsDir);
    Object inspectDataset(File datasetPath);
    void applyDatasetReview(UnetTrainPanel panel, Object review);
    DenseSegmentationTrainingConfig createTrainingConfig(UnetTrainPanel panel,
            String modelsDir, String device);
    DenseSegmentationInferenceService createInferenceService();
    DenseSegmentationTrainingService createTrainingService();
    String bestCheckpointName();
    String trainingConfigSummary(DenseSegmentationTrainingConfig config);
}
