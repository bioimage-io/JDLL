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
import java.io.IOException;
import java.nio.file.Files;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;
import org.apposed.appose.Service;
import org.apposed.appose.TaskException;

import io.bioimage.modelrunner.gui.custom.interfaces.ModelInstaller;
import io.bioimage.modelrunner.model.special.yolo.Yolo;
import io.bioimage.modelrunner.model.special.yolo.YoloTrainingProgress;
import io.bioimage.modelrunner.model.special.yolo.YoloValidationPreview;

public class YoloTrainingService {

    private final ModelInstaller installer;
    private File cancelSignalFile;
    private boolean cancelRequested;
    private Service runningPython;

    public YoloTrainingService(ModelInstaller installer) {
        this.installer = installer;
    }

    public void train(YoloTrainingConfig config,
            Consumer<YoloTrainingProgress> progressConsumer,
            Consumer<YoloValidationPreview> previewConsumer,
            Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException, TaskException {
        validate(config);
        File datasetYaml = YoloDatasetPreparer.prepare(config.getDatasetYamlPath(), config.getModelName(),
                config.getModelsDir(), logConsumer);
        if (!installer.isEnvironmentInstalled()) {
            installer.installEnvironment(logConsumer);
        }
        if (config.isFineTune() && !installer.isModelInstalled(config.getBaseModelPath())) {
            installer.installModelWeights(config.getBaseModelPath(), logConsumer);
        }
        File cancelFile = beginCancelSignal();
        try {
            Yolo.train(config.getEpochs(), config.getBaseModelPath(), config.getScratchArchitecture(),
                    datasetYaml.getAbsolutePath(),
                    config.getOutputWeightsPath(), config.getImageSize(), config.getPreviewEpochPeriod(),
                    progressConsumer, previewConsumer, logConsumer, config.getDevice(),
                    cancelFile.getAbsolutePath(), this::setRunningPython);
        } finally {
            finishCancelSignal(cancelFile);
        }
    }

    public synchronized void requestCancel() {
        cancelRequested = true;
        if (cancelSignalFile != null) {
            try {
                cancelSignalFile.createNewFile();
            } catch (IOException e) {
                // If the signal cannot be written, close the worker process below.
            }
        }
        Service python = runningPython;
        if (python != null && python.isAlive()) {
            python.close();
        }
    }

    public void close() {
        requestCancel();
    }

    private static void validate(YoloTrainingConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("Training configuration cannot be null.");
        }
        if (config.getModelName() == null || config.getModelName().trim().isEmpty()) {
            throw new IllegalArgumentException("Please provide a name for the YOLO model.");
        }
        if (config.getModelName().contains("/") || config.getModelName().contains("\\")
                || config.getModelName().contains("..")) {
            throw new IllegalArgumentException("The YOLO model name cannot contain path separators or '..'.");
        }
        if (config.getDatasetYamlPath() == null || config.getDatasetYamlPath().trim().isEmpty()) {
            throw new IllegalArgumentException("Please provide the YOLO training dataset path.");
        }
        File dataset = new File(config.getDatasetYamlPath());
        if (!dataset.exists()) {
            throw new IllegalArgumentException("The training dataset path does not exist: "
                    + config.getDatasetYamlPath());
        }
        if (config.getEpochs() <= 0) {
            throw new IllegalArgumentException("The number of epochs must be greater than zero.");
        }
        if (config.getImageSize() <= 0) {
            throw new IllegalArgumentException("The YOLO image size must be greater than zero.");
        }
        if (config.isFineTune() && (config.getBaseModelPath() == null || config.getBaseModelPath().trim().isEmpty())) {
            throw new IllegalArgumentException("Please select a base YOLO model for fine tuning.");
        }
        if (!config.isFineTune() && !YoloModelRegistry.isKnownScratchArchitecture(config.getScratchArchitecture())) {
            throw new IllegalArgumentException("Please select a valid YOLO architecture for training from scratch.");
        }
    }

    private synchronized File beginCancelSignal() throws IOException {
        cancelRequested = false;
        File signal = File.createTempFile("jdll-yolo-cancel-", ".flag");
        Files.deleteIfExists(signal.toPath());
        signal.deleteOnExit();
        cancelSignalFile = signal;
        return signal;
    }

    private synchronized void finishCancelSignal(File signal) {
        if (cancelSignalFile == signal) {
            cancelSignalFile = null;
        }
        cancelRequested = false;
        if (signal == null) {
            return;
        }
        try {
            Files.deleteIfExists(signal.toPath());
        } catch (IOException e) {
            // Best-effort cleanup of an out-of-process cancellation signal.
        }
    }

    private synchronized void setRunningPython(Service python) {
        runningPython = python;
    }
}
