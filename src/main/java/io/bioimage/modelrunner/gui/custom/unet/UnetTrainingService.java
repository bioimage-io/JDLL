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
package io.bioimage.modelrunner.gui.custom.unet;

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
import io.bioimage.modelrunner.model.special.unet.Unet;
import io.bioimage.modelrunner.model.special.unet.UnetTrainingProgress;
import io.bioimage.modelrunner.model.special.unet.UnetValidationPreview;
import io.bioimage.modelrunner.utils.JSONUtils;

public class UnetTrainingService {

    private final ModelInstaller installer;
    private Service runningPython;

    /**
     * Creates a new UnetTrainingService instance.
     *
     * @param installer the installer.
     */
    public UnetTrainingService(ModelInstaller installer) {
        this.installer = installer;
    }

    /**
     * Runs model training.
     *
     * @param config the training config.
     * @param progressConsumer the progress consumer.
     * @param logConsumer the log consumer.
     * @throws IOException if I/O fails.
     * @throws ExecutionException if install fails.
     * @throws InterruptedException if interrupted.
     * @throws BuildException if environment build fails.
     * @throws TaskException if the task fails.
     */
    public void train(UnetTrainingConfig config,
            Consumer<UnetTrainingProgress> progressConsumer,
            Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException, TaskException {
        train(config, progressConsumer, null, logConsumer);
    }

    /**
     * Runs model training.
     *
     * @param config the training config.
     * @param progressConsumer the progress consumer.
     * @param previewConsumer the validation preview consumer.
     * @param logConsumer the log consumer.
     * @throws IOException if I/O fails.
     * @throws ExecutionException if install fails.
     * @throws InterruptedException if interrupted.
     * @throws BuildException if environment build fails.
     * @throws TaskException if the task fails.
     */
    public void train(UnetTrainingConfig config,
            Consumer<UnetTrainingProgress> progressConsumer,
            Consumer<UnetValidationPreview> previewConsumer,
            Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException, TaskException {
        validate(config);
        if (!installer.isEnvironmentInstalled()) {
            installer.installEnvironment(logConsumer);
        }
        if (config.isFineTune() && !installer.isModelInstalled(config.getBaseModelPath())) {
            installer.installModelWeights(config.getBaseModelPath(), logConsumer);
        }
        Unet.train(toPythonConfig(config), progressConsumer, previewConsumer, logConsumer, this::setRunningPython);
    }

    /**
     * Requests cancellation.
     */
    public synchronized void requestCancel() {
        killRunningPython();
    }

    /**
     * Closes resources.
     */
    public void close() {
        requestCancel();
    }

    private static void validate(UnetTrainingConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("Training configuration cannot be null.");
        }
        if (config.getModelName() == null || config.getModelName().trim().isEmpty()) {
            throw new IllegalArgumentException("Please provide a name for the UNet model.");
        }
        if (config.getModelName().contains("/") || config.getModelName().contains("\\")
                || config.getModelName().contains("..")) {
            throw new IllegalArgumentException("The UNet model name cannot contain path separators or '..'.");
        }
        if (config.getDatasetPath() == null || config.getDatasetPath().trim().isEmpty()) {
            throw new IllegalArgumentException("Please provide the UNet training dataset path.");
        }
        if (!new File(config.getDatasetPath()).exists()) {
            throw new IllegalArgumentException("The UNet training dataset path does not exist: "
                    + config.getDatasetPath());
        }
        if (config.getEpochs() <= 0) {
            throw new IllegalArgumentException("The number of epochs must be greater than zero.");
        }
        if (config.isFineTune() && (config.getBaseModelPath() == null
                || !UnetModelRegistry.isModelPath(config.getBaseModelPath()))) {
            throw new IllegalArgumentException("Please select a base UNet model for fine tuning.");
        }
        if (!config.isFineTune() && !UnetModelRegistry.isKnownScratchArchitecture(config.getScratchArchitecture())) {
            throw new IllegalArgumentException("Please select a valid UNet architecture for training.");
        }
    }

    private static Map<String, Object> toPythonConfig(UnetTrainingConfig config) {
        Map<String, Object> values = new LinkedHashMap<String, Object>();
        values.put("model_name", config.getModelName());
        values.put("output_dir", new File(config.getOutputModelDir()).getAbsolutePath());
        values.put("dataset_path", new File(config.getDatasetPath()).getAbsolutePath());
        values.put("starting_point", config.isFineTune() ? "fine_tune" : "scratch");
        if (config.isFineTune()) {
            values.put("base_model", new File(config.getBaseModelPath()).getAbsolutePath());
        }
        values.put("architecture", config.isFineTune()
                ? architectureFromBaseModel(config.getBaseModelPath())
                : config.getScratchArchitecture());
        values.put("device", config.getDevice());
        values.put("epochs", config.getEpochs());
        values.put("task", "auto");
        values.put("axes", "auto");
        values.put("input_channels", "auto");
        values.put("output_classes", "auto");
        values.put("patch_size", "auto");
        values.put("batch_size", "auto");
        values.put("learning_rate", "auto");
        values.put("foreground_oversampling", "auto");
        values.put("foreground_probability", "auto");
        values.put("augmentation_profile", "auto");
        values.put("mixed_precision", "auto");
        return values;
    }

    private static String architectureFromBaseModel(String baseModelPath) {
        if (baseModelPath == null || baseModelPath.trim().isEmpty()) {
            return "tiny-2d";
        }
        File path = new File(baseModelPath);
        File folder = path.isDirectory() ? path : path.getParentFile();
        File configFile = folder == null ? null : new File(folder, "config.json");
        if (configFile == null || !configFile.isFile()) {
            return "tiny-2d";
        }
        try {
            Map<String, Object> config = JSONUtils.load(configFile.getAbsolutePath());
            Object architecture = config.get("architecture");
            return architecture == null || architecture.toString().trim().isEmpty()
                    ? "tiny-2d"
                    : architecture.toString();
        } catch (IOException e) {
            return "tiny-2d";
        }
    }

    private synchronized void setRunningPython(Service python) {
        runningPython = python;
    }

    private synchronized void killRunningPython() {
        Service python = runningPython;
        runningPython = null;
        if (python != null && python.isAlive()) {
            python.kill();
        }
    }
}
