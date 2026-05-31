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

    private static final long CANCEL_FALLBACK_TIMEOUT_MS = 250L;

    private final ModelInstaller installer;
    private File cancelSignalFile;
    private Service runningPython;

    /**
     * Creates a new YoloTrainingService instance.
     *
     * @param installer the installer.
     */
    public YoloTrainingService(ModelInstaller installer) {
        this.installer = installer;
    }

    /**
     * Runs model training.
     *
     * @param config the config.
     * @param progressConsumer the progress consumer callback.
     * @param previewConsumer the preview consumer callback.
     * @param logConsumer the log consumer callback.
     * @throws IOException if an I/O error occurs.
     * @throws ExecutionException if an asynchronous operation fails.
     * @throws InterruptedException if the current thread is interrupted.
     * @throws BuildException if the Python environment or service cannot be built.
     * @throws TaskException if task occurs.
     */
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

    /**
     * Performs request cancel.
     */
    public synchronized void requestCancel() {
        if (cancelSignalFile != null) {
            try {
                cancelSignalFile.createNewFile();
            } catch (IOException e) {
                killRunningPython();
                return;
            }
            killRunningPythonIfStillAliveLater();
        } else {
            killRunningPython();
        }
    }

    /**
     * Closes resources held by this object.
     */
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

    private void killRunningPythonIfStillAliveLater() {
        Service python = runningPython;
        if (python == null) {
            return;
        }
        Thread fallback = new Thread(() -> {
            try {
                Thread.sleep(CANCEL_FALLBACK_TIMEOUT_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
            synchronized (YoloTrainingService.this) {
                if (python.isAlive()) {
                    if (runningPython == python) {
                        runningPython = null;
                    }
                    python.kill();
                }
            }
        }, "yolo-training-cancel-fallback");
        fallback.setDaemon(true);
        fallback.start();
    }

    private synchronized void killRunningPython() {
        Service python = runningPython;
        runningPython = null;
        if (python != null && python.isAlive()) {
            python.kill();
        }
    }
}
