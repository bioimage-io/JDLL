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
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;
import org.apposed.appose.Service;
import org.apposed.appose.TaskException;

import io.bioimage.modelrunner.gui.custom.interfaces.ModelInstaller;
import io.bioimage.modelrunner.model.special.stardist.StarDist;
import io.bioimage.modelrunner.model.special.stardist.StardistTrainingProgress;
import io.bioimage.modelrunner.model.special.stardist.StardistValidationPreview;

public class StardistTrainingService {

    private final ModelInstaller installer;
    private Service runningPython;

    public StardistTrainingService(ModelInstaller installer) {
        this.installer = installer;
    }

    public void train(StardistTrainingConfig config,
            Consumer<StardistTrainingProgress> progressConsumer,
            Consumer<StardistValidationPreview> previewConsumer,
            Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException, TaskException {
        validate(config);
        File datasetRoot = new File(config.getDatasetPath());

        if (!installer.isEnvironmentInstalled()) {
            installer.installEnvironment(logConsumer);
        }
        if (config.isFineTune()) {
            throw new IllegalArgumentException("StarDist fine tuning is not wired yet. Use train from scratch.");
        }

        Map<String, Object> trainingConfig =
                new LinkedHashMap<String, Object>(StarDist.defaultTrainingConfig(config.getEpochs()));
        applyArchitectureDefaults(trainingConfig, config.getScratchArchitecture());
        StarDist.train(datasetRoot.getAbsolutePath(), null,
                config.getOutputModelDir(), config.getDevice(), config.getImageChannels(),
                config.getLabelColorMode(), config.getValidFraction(), trainingConfig,
                progressConsumer, previewConsumer, logConsumer, this::setRunningPython);
    }

    public synchronized void requestCancel() {
        Service python = runningPython;
        if (python != null && python.isAlive()) {
            python.close();
        }
    }

    public void close() {
        requestCancel();
    }

    public static boolean isValidDatasetPath(File datasetPath) {
        return datasetPath != null && datasetPath.isDirectory() && datasetPath.canRead();
    }

    private static void validate(StardistTrainingConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("Training configuration cannot be null.");
        }
        if (config.getModelName() == null || config.getModelName().trim().isEmpty()) {
            throw new IllegalArgumentException("Please provide a name for the StarDist model.");
        }
        if (config.getModelName().contains("/") || config.getModelName().contains("\\")
                || config.getModelName().contains("..")) {
            throw new IllegalArgumentException("The StarDist model name cannot contain path separators or '..'.");
        }
        if (config.getDatasetPath() == null || config.getDatasetPath().trim().isEmpty()) {
            throw new IllegalArgumentException("Please provide the StarDist training dataset path.");
        }
        if (!isValidDatasetPath(new File(config.getDatasetPath()))) {
            throw new IllegalArgumentException("Please select a readable StarDist dataset folder.");
        }
        if (config.getEpochs() <= 0) {
            throw new IllegalArgumentException("The number of epochs must be greater than zero.");
        }
        if (!config.isFineTune()
                && !StardistModelRegistry.isKnownScratchArchitecture(config.getScratchArchitecture())) {
            throw new IllegalArgumentException("Please select a valid StarDist architecture for training.");
        }
    }

    private static void applyArchitectureDefaults(Map<String, Object> trainingConfig, String architecture) {
        if (architecture == null) {
            return;
        }
        String arch = architecture.toLowerCase();
        if (arch.contains("small")) {
            trainingConfig.put("train_patch_size", java.util.Arrays.asList(192, 192));
            trainingConfig.put("train_batch_size", 8);
        } else if (arch.contains("big")) {
            trainingConfig.put("train_patch_size", java.util.Arrays.asList(384, 384));
            trainingConfig.put("train_batch_size", 2);
            trainingConfig.put("n_rays", 64);
        }
    }

    private synchronized void setRunningPython(Service python) {
        runningPython = python;
    }
}
