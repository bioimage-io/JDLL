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

/** UNet behavior for the shared dense-segmentation interface. */
public final class UnetBackend implements DenseSegmentationBackend {

    @Override public String getDisplayName() { return "UNet"; }
    @Override public String getMaskSuffix() { return "_unet_labels"; }
    @Override public String getModelFileDescription() { return "UNet weights (*.pt, *.pth)"; }
    @Override public String[] getModelFileExtensions() { return new String[] {"pt", "pth"}; }

    @Override
    public LinkedHashMap<String, String> buildModelEntries(String modelsDir) {
        return UnetModelRegistry.buildModelEntries(modelsDir);
    }

    @Override
    public File findModelFile(File path) {
        return path != null && path.isFile() ? path : UnetModelRegistry.findModelFile(path);
    }

    @Override
    public String modelLabel(File selected) {
        return "[Custom] " + (selected.isDirectory() ? selected.getName()
                : UnetModelRegistry.removeWeightsExtension(selected.getName()));
    }

    @Override
    public void configureTrainingPanel(UnetTrainPanel panel, String modelsDir) {
        panel.setModelsDir(modelsDir);
    }

    @Override
    public void refreshTrainingPanel(UnetTrainPanel panel, String modelsDir) {
        panel.setModelsDir(modelsDir);
        panel.refreshScratchArchitectures();
    }

    @Override
    public Object inspectDataset(File datasetPath) {
        return UnetDatasetInspector.inspect(datasetPath);
    }

    @Override
    public void applyDatasetReview(UnetTrainPanel panel, Object review) {
        panel.setDatasetDimensionality((UnetDatasetInspector.Dimensionality) review);
    }

    @Override
    public DenseSegmentationTrainingConfig createTrainingConfig(UnetTrainPanel panel,
            String modelsDir, String device) {
        return UnetTrainingConfig.fromUi(panel.getModelNameField().getText(),
                panel.getDatasetField().getText(),
                Integer.parseInt(panel.getEpochsField().getText().trim()),
                panel.getFineTuneRadio().isSelected(), panel.getSelectedBaseModelValue(),
                panel.getSelectedScratchArchitectureValue(), modelsDir, device);
    }

    @Override public DenseSegmentationInferenceService createInferenceService() {
        return new UnetInferenceService(new UnetInstaller());
    }

    @Override public DenseSegmentationTrainingService createTrainingService() {
        return new UnetTrainingService(new UnetInstaller());
    }

    @Override public String bestCheckpointName() { return "weights_best.pt"; }

    @Override
    public String trainingConfigSummary(DenseSegmentationTrainingConfig config) {
        String start = config.isFineTune() ? "fine tune from " + config.getBaseModelPath()
                : "train from scratch using " + config.getScratchArchitecture();
        return "epochs=" + config.getEpochs() + ", device=" + config.getDevice()
                + ", starting_point=" + start;
    }
}
