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
import java.util.LinkedHashMap;

import io.bioimage.modelrunner.gui.custom.unet.DenseSegmentationBackend;
import io.bioimage.modelrunner.gui.custom.unet.DenseSegmentationInferenceService;
import io.bioimage.modelrunner.gui.custom.unet.DenseSegmentationTrainingConfig;
import io.bioimage.modelrunner.gui.custom.unet.DenseSegmentationTrainingService;
import io.bioimage.modelrunner.gui.custom.unet.UnetTrainPanel;

/** Cross-GOOSE behavior for the shared dense-segmentation interface. */
public final class CrossGooseBackend implements DenseSegmentationBackend {

    @Override public String getDisplayName() { return "Cross-GOOSE"; }
    @Override public String getMaskSuffix() { return "_cross_goose_labels"; }
    @Override public String getModelFileDescription() { return "Cross-GOOSE checkpoints (*.ckpt)"; }
    @Override public String[] getModelFileExtensions() { return new String[] {"ckpt"}; }

    @Override
    public LinkedHashMap<String, String> buildModelEntries(String modelsDir) {
        return CrossGooseModelRegistry.buildModelEntries(modelsDir);
    }

    @Override public File findModelFile(File path) { return CrossGooseModelRegistry.modelDirectory(path); }

    @Override
    public String modelLabel(File selected) {
        File directory = CrossGooseModelRegistry.modelDirectory(selected);
        return "[Custom] " + (directory == null ? selected.getName() : directory.getName());
    }

    @Override
    public void configureTrainingPanel(UnetTrainPanel panel, String modelsDir) {
        panel.configureFixedArchitectures(CrossGooseModelRegistry.scratchArchitectures(),
                CrossGooseModelRegistry.DEFAULT_ARCHITECTURE,
                CrossGooseModelRegistry::isModelPath,
                CrossGooseModelRegistry::isKnownScratchArchitecture,
                getModelFileDescription(), getModelFileExtensions());
    }

    @Override public void refreshTrainingPanel(UnetTrainPanel panel, String modelsDir) {
        panel.refreshScratchArchitectures();
    }

    @Override public Object inspectDataset(File datasetPath) { return Boolean.TRUE; }
    @Override public void applyDatasetReview(UnetTrainPanel panel, Object review) { }

    @Override
    public DenseSegmentationTrainingConfig createTrainingConfig(UnetTrainPanel panel,
            String modelsDir, String device) {
        return CrossGooseTrainingConfig.fromUi(panel.getModelNameField().getText(),
                panel.getDatasetField().getText(),
                Integer.parseInt(panel.getEpochsField().getText().trim()),
                panel.getFineTuneRadio().isSelected(), panel.getSelectedBaseModelValue(),
                panel.getSelectedScratchArchitectureValue(), modelsDir, device);
    }

    @Override public DenseSegmentationInferenceService createInferenceService() {
        return new CrossGooseInferenceService(new CrossGooseInstaller());
    }

    @Override public DenseSegmentationTrainingService createTrainingService() {
        return new CrossGooseTrainingService(new CrossGooseInstaller());
    }

    @Override public String bestCheckpointName() { return "checkpoints/best.ckpt"; }

    @Override
    public String trainingConfigSummary(DenseSegmentationTrainingConfig config) {
        String start = config.isFineTune() ? "fine tune from " + config.getBaseModelPath()
                : "train from scratch using the default Cross-GOOSE architecture";
        return "epochs=" + config.getEpochs() + ", device=" + config.getDevice()
                + ", starting_point=" + start;
    }
}
