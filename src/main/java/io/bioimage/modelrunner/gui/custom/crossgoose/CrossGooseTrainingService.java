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
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;
import org.apposed.appose.Service;
import org.apposed.appose.TaskException;

import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer;
import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer.PreparedDataset;
import io.bioimage.modelrunner.gui.custom.unet.DenseSegmentationTrainingConfig;
import io.bioimage.modelrunner.gui.custom.unet.DenseSegmentationTrainingService;
import io.bioimage.modelrunner.model.special.crossgoose.CrossGoose;
import io.bioimage.modelrunner.model.special.unet.UnetTrainingProgress;
import io.bioimage.modelrunner.model.special.unet.UnetValidationPreview;

/** Prepares datasets and launches Cross-GOOSE training. */
public final class CrossGooseTrainingService implements DenseSegmentationTrainingService {

    private final CrossGooseInstaller installer;
    private Service runningPython;

    public CrossGooseTrainingService(CrossGooseInstaller installer) {
        this.installer = installer;
    }

    @Override
    public void train(DenseSegmentationTrainingConfig values,
            Consumer<UnetTrainingProgress> progressConsumer,
            Consumer<UnetValidationPreview> previewConsumer, Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException, TaskException {
        if (!(values instanceof CrossGooseTrainingConfig)) {
            throw new IllegalArgumentException("Expected a Cross-GOOSE training configuration.");
        }
        CrossGooseTrainingConfig config = (CrossGooseTrainingConfig) values;
        validate(config);
        if (!installer.isEnvironmentInstalled()) installer.installEnvironment(logConsumer);
        if (config.isFineTune() && !installer.isModelInstalled(config.getBaseModelPath())) {
            installer.installModelWeights(config.getBaseModelPath(), logConsumer);
        }
        PreparedDataset dataset = SegmentationDatasetPreparer.prepare(config.getDatasetPath(),
                config.getModelName(), config.getModelsDir(), 0.15d,
                SegmentationDatasetPreparer.Framework.UNET, logConsumer);
        CrossGoose.train(pythonConfig(config, dataset.getDatasetRoot()), progressConsumer,
                previewConsumer, logConsumer, this::setRunningPython);
    }

    @Override public void close() {
        Service python;
        synchronized (this) {
            python = runningPython;
            runningPython = null;
        }
        if (python != null && python.isAlive()) python.kill();
    }

    private static void validate(CrossGooseTrainingConfig config) {
        if (config.getModelName() == null || config.getModelName().trim().isEmpty()) {
            throw new IllegalArgumentException("Please provide a name for the Cross-GOOSE model.");
        }
        if (config.getDatasetPath() == null || !new File(config.getDatasetPath()).exists()) {
            throw new IllegalArgumentException("Cross-GOOSE dataset path does not exist: "
                    + config.getDatasetPath());
        }
        if (config.getEpochs() <= 0) throw new IllegalArgumentException("Epochs must be positive.");
        if (config.isFineTune() && !CrossGooseModelRegistry.isModelPath(config.getBaseModelPath())) {
            throw new IllegalArgumentException("Please select a valid Cross-GOOSE base model.");
        }
        if (!config.isFineTune()
                && !CrossGooseModelRegistry.isKnownScratchArchitecture(config.getScratchArchitecture())) {
            throw new IllegalArgumentException("Please select the default Cross-GOOSE architecture.");
        }
    }

    private static Map<String, Object> pythonConfig(CrossGooseTrainingConfig config, File dataset) {
        Map<String, Object> values = new LinkedHashMap<String, Object>();
        values.put("model_name", config.getModelName());
        values.put("output_dir", config.getOutputModelDir());
        values.put("dataset_path", dataset.getAbsolutePath());
        values.put("architecture", CrossGooseModelRegistry.DEFAULT_ARCHITECTURE);
        values.put("starting_point", config.isFineTune() ? "fine_tune" : "scratch");
        if (config.isFineTune()) values.put("base_model", config.getBaseModelPath());
        values.put("device", config.getDevice());
        values.put("epochs", config.getEpochs());
        return values;
    }

    private synchronized void setRunningPython(Service service) { runningPython = service; }
}
