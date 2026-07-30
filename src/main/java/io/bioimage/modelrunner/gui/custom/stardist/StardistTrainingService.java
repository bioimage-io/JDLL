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
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;
import org.apposed.appose.Service;
import org.apposed.appose.TaskException;

import io.bioimage.modelrunner.gui.custom.interfaces.ModelInstaller;
import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer;
import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer.Dimensionality;
import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer.PreparedDataset;
import io.bioimage.modelrunner.gui.custom.stardist.StardistModelRegistry.FineTuneSource;
import io.bioimage.modelrunner.model.special.stardist.StarDist;
import io.bioimage.modelrunner.model.special.stardist.StardistTrainingProgress;
import io.bioimage.modelrunner.model.special.stardist.StardistValidationPreview;

public class StardistTrainingService {

    private static final long CANCEL_FALLBACK_TIMEOUT_MS = 250L;
    private static final double FINE_TUNE_LEARNING_RATE = 1e-4d;
    private static final String INITIAL_WEIGHTS_NAME = "weights_initial.h5";
    private static final String INITIAL_CONFIG_NAME = "config_initial.json";
    private static final String INITIAL_THRESHOLDS_NAME = "thresholds_initial.json";

    private final ModelInstaller installer;
    private File cancelSignalFile;
    private Service runningPython;

    /**
     * Creates a new StardistTrainingService instance.
     *
     * @param installer the installer.
     */
    public StardistTrainingService(ModelInstaller installer) {
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
    public void train(StardistTrainingConfig config,
            Consumer<StardistTrainingProgress> progressConsumer,
            Consumer<StardistValidationPreview> previewConsumer,
            Consumer<String> logConsumer)
            throws IOException, ExecutionException, InterruptedException, BuildException, TaskException {
        validate(config);
        PreparedDataset dataset = SegmentationDatasetPreparer.prepare(config.getDatasetPath(), config.getModelName(),
                config.getModelsDir(), config.getValidFraction(), SegmentationDatasetPreparer.Framework.STARDIST,
                logConsumer);
        File datasetRoot = dataset.getDatasetRoot();

        if (!installer.isEnvironmentInstalled()) {
            installer.installEnvironment(logConsumer);
        }

        Map<String, Object> trainingConfig =
                new LinkedHashMap<String, Object>(StarDist.defaultTrainingConfig(config.getEpochs()));
        if (config.isFineTune()) {
            if (!installer.isModelInstalled(config.getBaseModelPath())) {
                installer.installModelWeights(config.getBaseModelPath(), logConsumer);
            }
            FineTuneSource source = StardistModelRegistry.resolveFineTuneSource(config.getBaseModelPath());
            int datasetDimensions = dataset.getDimensionality().getDimensions();
            if (source == null || StardistModelRegistry.configDimensions(source.getConfig()) != datasetDimensions) {
                throw new IllegalArgumentException("The selected StarDist fine-tuning model requires "
                        + "compatible HDF5 weights and a " + datasetDimensions + "D config.json.");
            }
            prepareFineTuning(trainingConfig, config, source, dataset, logConsumer);
        } else {
            if (StardistModelRegistry.isArchitecture3D(config.getScratchArchitecture()) != dataset.is3D()) {
                throw new IllegalArgumentException("The selected StarDist architecture does not match the "
                        + dataset.getDimensionality().getDimensions() + "D dataset.");
            }
            applyArchitectureDefaults(trainingConfig, config.getScratchArchitecture());
        }
        applyDimensionalityDefaults(trainingConfig, dataset.getDimensionality());
        trainingConfig.put("train_epochs", config.getEpochs());
        File cancelFile = beginCancelSignal();
        try {
            StarDist.train(datasetRoot.getAbsolutePath(), null,
                    config.getOutputModelDir(), config.getDevice(), dataset.getImageChannels(),
                    config.getLabelColorMode(), config.getValidFraction(), trainingConfig,
                    progressConsumer, previewConsumer, logConsumer, cancelFile.getAbsolutePath(), this::setRunningPython);
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
        }
        killRunningPythonIfStillAliveLater();
    }

    /**
     * Closes resources held by this object.
     */
    public void close() {
        requestCancel();
    }

    /**
     * Returns whether valid dataset path.
     *
     * @param datasetPath the dataset path.
     * @return true if valid dataset path; false otherwise.
     */
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
        if (config.isFineTune()
                && !StardistModelRegistry.isSelectableFineTuneSource(config.getBaseModelPath())) {
            throw new IllegalArgumentException("Please select a StarDist model with HDF5 weights and config.json.");
        }
    }

    private static void prepareFineTuning(Map<String, Object> trainingConfig, StardistTrainingConfig config,
            FineTuneSource source, PreparedDataset dataset, Consumer<String> logConsumer) throws IOException {
        Map<String, Object> sourceConfig = sanitizedModelConfig(source.getConfig());
        trainingConfig.putAll(sourceConfig);
        trainingConfig.put("train_epochs", config.getEpochs());
        trainingConfig.put("train_steps_per_epoch", 100);
        trainingConfig.put("train_learning_rate", FINE_TUNE_LEARNING_RATE);
        if (dataset.is3D()) {
            trainingConfig.put("train_batch_size", 1);
            trainingConfig.putIfAbsent("train_patch_size", java.util.Arrays.asList(32, 128, 128));
            trainingConfig.put("validation_preview_count", 1);
        } else {
            trainingConfig.put("train_batch_size", 4);
            trainingConfig.put("train_patch_size", java.util.Arrays.asList(256, 256));
            trainingConfig.put("validation_preview_count", 20);
        }
        trainingConfig.put("train_tensorboard", false);
        trainingConfig.put("train_checkpoint", "weights_best.h5");
        trainingConfig.put("train_checkpoint_last", "weights_last.h5");

        File output = new File(config.getOutputModelDir());
        Files.createDirectories(output.toPath());
        File initialWeights = new File(output, INITIAL_WEIGHTS_NAME);
        Files.copy(source.getWeightsFile().toPath(), initialWeights.toPath(),
                StandardCopyOption.COPY_ATTRIBUTES);
        Files.copy(source.getConfigFile().toPath(), new File(output, INITIAL_CONFIG_NAME).toPath(),
                StandardCopyOption.COPY_ATTRIBUTES);
        File sourceThresholds = new File(source.getModelDirectory(), "thresholds.json");
        if (sourceThresholds.isFile()) {
            Files.copy(sourceThresholds.toPath(), new File(output, INITIAL_THRESHOLDS_NAME).toPath(),
                    StandardCopyOption.COPY_ATTRIBUTES);
        }

        int sourceChannels = intValue(sourceConfig.get("n_channel_in"), 1);
        trainingConfig.put("_jdll_fine_tune_weights", initialWeights.getAbsolutePath());
        trainingConfig.put("_jdll_fine_tune_source_config", sourceConfig);
        log(logConsumer, "Fine-tuning source: " + source.getModelDirectory().getAbsolutePath());
        log(logConsumer, "Copied initial weights to: " + initialWeights.getAbsolutePath());
        log(logConsumer, "Fine-tuning configuration: learning_rate=" + FINE_TUNE_LEARNING_RATE
                + ", fresh_optimizer=true, frozen_layers=0, source_channels=" + sourceChannels
                + ", training_channels=" + dataset.getTargetImageChannels() + ".");
        if (sourceChannels != dataset.getTargetImageChannels()) {
            log(logConsumer, "The first StarDist convolution will be adapted from " + sourceChannels
                    + " to " + dataset.getTargetImageChannels() + " input channels.");
        }
    }

    private static Map<String, Object> sanitizedModelConfig(Map<String, Object> source) {
        Map<String, Object> config = new LinkedHashMap<String, Object>();
        for (Map.Entry<String, Object> entry : source.entrySet()) {
            if (!isStardistMetadataKey(entry.getKey())) {
                config.put(entry.getKey(), entry.getValue());
            }
        }
        config.remove("n_dim");
        config.remove("n_channel_out");
        config.remove("net_input_shape");
        config.remove("net_mask_shape");
        return config;
    }

    private static int intValue(Object value, int fallback) {
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        try {
            return value == null ? fallback : Integer.parseInt(value.toString());
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static void log(Consumer<String> consumer, String message) {
        if (consumer != null) {
            consumer.accept(message);
        }
    }

    private static void applyArchitectureDefaults(Map<String, Object> trainingConfig, String architecture) {
        if (architecture == null) {
            return;
        }
        Map<String, Object> customConfig = StardistModelRegistry.loadCustomScratchConfig(architecture);
        if (customConfig != null) {
            for (Map.Entry<String, Object> entry : customConfig.entrySet()) {
                if (!isStardistMetadataKey(entry.getKey())) {
                    trainingConfig.put(entry.getKey(), entry.getValue());
                }
            }
            return;
        }
        String arch = architecture.toLowerCase();
        boolean volume = StardistModelRegistry.isArchitecture3D(architecture);
        if (volume && (arch.contains("medium_big") || arch.contains("medium-big"))) {
            apply3dPreset(trainingConfig, 3, 24, 96, 32, 160, 160);
        } else if (volume && arch.contains("small")) {
            apply3dPreset(trainingConfig, 2, 16, 64, 32, 96, 96);
        } else if (volume && arch.contains("medium")) {
            apply3dPreset(trainingConfig, 2, 32, 128, 32, 128, 128);
        } else if (volume && arch.contains("large")) {
            apply3dPreset(trainingConfig, 3, 48, 192, 64, 192, 192);
        } else if (volume && arch.contains("big")) {
            apply3dPreset(trainingConfig, 3, 32, 128, 48, 160, 160);
        } else if (arch.contains("small")) {
            trainingConfig.put("unet_n_depth", 3);
            trainingConfig.put("unet_n_filter_base", 16);
            trainingConfig.put("net_conv_after_unet", 64);
            trainingConfig.put("train_patch_size", java.util.Arrays.asList(192, 192));
            trainingConfig.put("train_batch_size", 8);
        } else if (arch.contains("medium_big") || arch.contains("medium-big")) {
            trainingConfig.put("unet_n_depth", 4);
            trainingConfig.put("unet_n_filter_base", 24);
            trainingConfig.put("net_conv_after_unet", 96);
            trainingConfig.put("train_patch_size", java.util.Arrays.asList(320, 320));
            trainingConfig.put("train_batch_size", 3);
        } else if (arch.contains("medium")) {
            trainingConfig.put("unet_n_depth", 3);
            trainingConfig.put("unet_n_filter_base", 32);
            trainingConfig.put("net_conv_after_unet", 128);
            trainingConfig.put("train_patch_size", java.util.Arrays.asList(256, 256));
            trainingConfig.put("train_batch_size", 4);
        } else if (arch.contains("big")) {
            trainingConfig.put("unet_n_depth", 4);
            trainingConfig.put("unet_n_filter_base", 32);
            trainingConfig.put("net_conv_after_unet", 128);
            trainingConfig.put("train_patch_size", java.util.Arrays.asList(384, 384));
            trainingConfig.put("train_batch_size", 2);
        } else if (arch.contains("large")) {
            trainingConfig.put("unet_n_depth", 4);
            trainingConfig.put("unet_n_filter_base", 48);
            trainingConfig.put("net_conv_after_unet", 192);
            trainingConfig.put("train_patch_size", java.util.Arrays.asList(512, 512));
            trainingConfig.put("train_batch_size", 1);
        }
    }

    private static void apply3dPreset(Map<String, Object> config, int depth, int filters,
            int finalFilters, int z, int y, int x) {
        config.put("unet_n_depth", depth);
        config.put("unet_n_filter_base", filters);
        config.put("net_conv_after_unet", finalFilters);
        config.put("n_rays", 96);
        config.put("train_patch_size", java.util.Arrays.asList(z, y, x));
        config.put("train_batch_size", 1);
        config.put("validation_preview_count", 1);
    }

    private static void applyDimensionalityDefaults(Map<String, Object> config, Dimensionality dimensionality) {
        if (dimensionality == Dimensionality.THREE_D) {
            config.put("n_dim", 3);
            config.put("axes", "ZYXC");
            config.put("n_rays", intValue(config.get("n_rays"), 96));
            if (!(config.get("grid") instanceof java.util.List<?>)
                    || ((java.util.List<?>) config.get("grid")).size() != 3) {
                config.put("grid", java.util.Arrays.asList(1, 1, 1));
            }
            config.put("_jdll_auto_anisotropy", config.get("anisotropy") == null);
        } else {
            config.put("n_dim", 2);
            config.put("axes", "YXC");
            config.putIfAbsent("grid", java.util.Arrays.asList(1, 1));
        }
    }

    private static boolean isStardistMetadataKey(String key) {
        return "framework".equals(key) || "format".equals(key) || "format_version".equals(key)
                || "model".equals(key) || "dataset".equals(key) || "outputs".equals(key)
                || "logging".equals(key);
    }

    private synchronized void setRunningPython(Service python) {
        runningPython = python;
    }

    private synchronized File beginCancelSignal() throws IOException {
        File signal = File.createTempFile("jdll-stardist-cancel-", ".flag");
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
            synchronized (StardistTrainingService.this) {
                if (python.isAlive()) {
                    if (runningPython == python) {
                        runningPython = null;
                    }
                    python.kill();
                }
            }
        }, "stardist-training-cancel-fallback");
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
