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
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import io.bioimage.modelrunner.gui.custom.training.TrainingModelPaths;
import io.bioimage.modelrunner.model.special.yolo.Yolo;

public final class YoloTrainingConfig {

    public static final int DEFAULT_IMAGE_SIZE = 640;
    public static final int DEFAULT_PREVIEW_EPOCH_PERIOD = 1;
    private static final Gson CONFIG_GSON = new GsonBuilder().setPrettyPrinting().create();

    private final String modelName;
    private final String datasetYamlPath;
    private final int epochs;
    private final int imageSize;
    private final boolean fineTune;
    private final String baseModelPath;
    private final String scratchArchitecture;
    private final String modelsDir;
    private final String outputWeightsPath;
    private final int previewEpochPeriod;
    private final String device;

    /**
     * Creates a new YoloTrainingConfig instance.
     *
     * @param modelName the model name.
     * @param datasetYamlPath the dataset YAML path.
     * @param epochs the epochs.
     * @param imageSize the image size.
     * @param fineTune the fine tune.
     * @param baseModelPath the base model path.
     * @param scratchArchitecture the scratch architecture.
     * @param modelsDir the models directory.
     * @param outputWeightsPath the output weights path.
     * @param previewEpochPeriod the preview epoch period.
     */
    public YoloTrainingConfig(String modelName, String datasetYamlPath, int epochs, int imageSize,
            boolean fineTune, String baseModelPath, String scratchArchitecture,
            String modelsDir, String outputWeightsPath, int previewEpochPeriod) {
        this(modelName, datasetYamlPath, epochs, imageSize, fineTune, baseModelPath, scratchArchitecture,
                modelsDir, outputWeightsPath, previewEpochPeriod, "cpu");
    }

    /**
     * Creates a new YoloTrainingConfig instance.
     *
     * @param modelName the model name.
     * @param datasetYamlPath the dataset YAML path.
     * @param epochs the epochs.
     * @param imageSize the image size.
     * @param fineTune the fine tune.
     * @param baseModelPath the base model path.
     * @param scratchArchitecture the scratch architecture.
     * @param modelsDir the models directory.
     * @param outputWeightsPath the output weights path.
     * @param previewEpochPeriod the preview epoch period.
     * @param device the device.
     */
    public YoloTrainingConfig(String modelName, String datasetYamlPath, int epochs, int imageSize,
            boolean fineTune, String baseModelPath, String scratchArchitecture,
            String modelsDir, String outputWeightsPath, int previewEpochPeriod, String device) {
        this.modelName = modelName;
        this.datasetYamlPath = datasetYamlPath;
        this.epochs = epochs;
        this.imageSize = imageSize;
        this.fineTune = fineTune;
        this.baseModelPath = baseModelPath;
        this.scratchArchitecture = scratchArchitecture;
        this.modelsDir = modelsDir;
        this.outputWeightsPath = outputWeightsPath;
        this.previewEpochPeriod = previewEpochPeriod;
        this.device = normalizeDevice(device);
    }

    /**
     * Returns the model name.
     *
     * @return the model name.
     */
    public String getModelName() {
        return modelName;
    }

    /**
     * Returns the dataset YAML path.
     *
     * @return the dataset YAML path.
     */
    public String getDatasetYamlPath() {
        return datasetYamlPath;
    }

    /**
     * Returns the epochs.
     *
     * @return the epochs.
     */
    public int getEpochs() {
        return epochs;
    }

    /**
     * Returns the image size.
     *
     * @return the image size.
     */
    public int getImageSize() {
        return imageSize;
    }

    /**
     * Returns whether fine tune.
     *
     * @return true if fine tune; false otherwise.
     */
    public boolean isFineTune() {
        return fineTune;
    }

    /**
     * Returns the base model path.
     *
     * @return the base model path.
     */
    public String getBaseModelPath() {
        return baseModelPath;
    }

    /**
     * Returns the scratch architecture.
     *
     * @return the scratch architecture.
     */
    public String getScratchArchitecture() {
        return scratchArchitecture;
    }

    /**
     * Returns the model source used for training.
     *
     * @return the model source.
     */
    public String getTrainingModelSource() {
        return modelSource();
    }

    /**
     * Returns the models directory.
     *
     * @return the models directory.
     */
    public String getModelsDir() {
        return modelsDir;
    }

    /**
     * Returns the output weights path.
     *
     * @return the output weights path.
     */
    public String getOutputWeightsPath() {
        return outputWeightsPath;
    }

    /**
     * Returns the preview epoch period.
     *
     * @return the preview epoch period.
     */
    public int getPreviewEpochPeriod() {
        return previewEpochPeriod;
    }

    /**
     * Returns the device.
     *
     * @return the device.
     */
    public String getDevice() {
        return device;
    }

    /**
     * Returns the output model directory.
     *
     * @return the output model directory.
     */
    public File getOutputModelDir() {
        File output = new File(outputWeightsPath).getAbsoluteFile();
        File parent = output.getParentFile();
        return parent == null ? new File(".").getAbsoluteFile() : parent;
    }

    /**
     * Returns the generated training config file.
     *
     * @return the config file.
     */
    public File getConfigFile() {
        return new File(getOutputModelDir(), "config.json");
    }

    /**
     * Writes a JSON training config file.
     *
     * @param resolvedDatasetYaml the actual data.yaml used by Ultralytics.
     * @return the written config file.
     * @throws IOException if an I/O error occurs.
     */
    public File writeConfig(File resolvedDatasetYaml) throws IOException {
        File configFile = getConfigFile();
        File parent = configFile.getParentFile();
        if (parent != null) {
            Files.createDirectories(parent.toPath());
        }
        try (Writer writer = Files.newBufferedWriter(configFile.toPath(), StandardCharsets.UTF_8)) {
            CONFIG_GSON.toJson(toConfigMap(resolvedDatasetYaml), writer);
        }
        return configFile;
    }

    /**
     * Formats all training parameters for the UI log.
     *
     * @param resolvedDatasetYaml the actual data.yaml used by Ultralytics.
     * @return the formatted parameter lines.
     */
    public List<String> toLogLines(File resolvedDatasetYaml) {
        List<String> lines = new ArrayList<String>();
        flatten("", toConfigMap(resolvedDatasetYaml), lines);
        return lines;
    }

    /**
     * Builds a JSON-serializable training config map.
     *
     * @param resolvedDatasetYaml the actual data.yaml used by Ultralytics.
     * @return the config map.
     */
    public Map<String, Object> toConfigMap(File resolvedDatasetYaml) {
        Map<String, Object> root = new LinkedHashMap<String, Object>();
        root.put("framework", "yolo");
        root.put("format_version", 1);
        root.put("model", modelMap());
        root.put("dataset", datasetMap(resolvedDatasetYaml));
        root.put("ultralytics_train", ultralyticsTrainMap(resolvedDatasetYaml));
        root.put("validation_preview", validationPreviewMap());
        root.put("logging", loggingMap());
        root.put("outputs", outputsMap());
        root.put("dataset_preparation", YoloDatasetPreparer.parameterSnapshot(imageSize));
        return root;
    }

    private Map<String, Object> modelMap() {
        Map<String, Object> map = new LinkedHashMap<String, Object>();
        map.put("name", modelName);
        map.put("start_mode", fineTune ? "fine_tune" : "from_scratch");
        map.put("fine_tune", fineTune);
        map.put("base_model_path", baseModelPath);
        map.put("scratch_architecture", scratchArchitecture);
        map.put("model_source", modelSource());
        map.put("models_dir", modelsDir);
        return map;
    }

    private Map<String, Object> datasetMap(File resolvedDatasetYaml) {
        Map<String, Object> map = new LinkedHashMap<String, Object>();
        map.put("requested_path", datasetYamlPath);
        map.put("resolved_yaml_path", absolutePath(resolvedDatasetYaml));
        return map;
    }

    private Map<String, Object> ultralyticsTrainMap(File resolvedDatasetYaml) {
        Map<String, Object> map = new LinkedHashMap<String, Object>();
        map.put("data", absolutePath(resolvedDatasetYaml));
        map.put("epochs", epochs);
        map.put("imgsz", imageSize);
        map.put("batch", Yolo.DEFAULT_TRAIN_BATCH_SIZE);
        map.put("project", projectDir().getAbsolutePath());
        map.put("name", runName());
        map.put("exist_ok", Yolo.DEFAULT_TRAIN_EXIST_OK);
        map.put("verbose", Yolo.DEFAULT_TRAIN_VERBOSE);
        map.put("plots", Yolo.DEFAULT_TRAIN_PLOTS);
        map.put("workers", Yolo.DEFAULT_TRAIN_WORKERS);
        map.put("requested_device", device);
        map.put("device", trainDevice());
        return map;
    }

    private Map<String, Object> validationPreviewMap() {
        Map<String, Object> map = new LinkedHashMap<String, Object>();
        map.put("epoch_period", previewEpochPeriod);
        map.put("sample_count", Yolo.DEFAULT_VALIDATION_PREVIEW_SAMPLE_COUNT);
        map.put("confidence_threshold", Yolo.DEFAULT_VALIDATION_PREVIEW_CONFIDENCE);
        map.put("directory", new File(runDir(), "validation_preview").getAbsolutePath());
        map.put("latest_json", new File(new File(runDir(), "validation_preview"), "latest.json").getAbsolutePath());
        return map;
    }

    private Map<String, Object> loggingMap() {
        boolean accelerated = !"cpu".equals(device);
        Map<String, Object> map = new LinkedHashMap<String, Object>();
        map.put("progress_every_n_steps", accelerated
                ? Yolo.DEFAULT_ACCELERATED_PROGRESS_EVERY_N_STEPS
                : Yolo.DEFAULT_CPU_PROGRESS_EVERY_N_STEPS);
        map.put("log_every_n_steps", accelerated
                ? Yolo.DEFAULT_ACCELERATED_LOG_EVERY_N_STEPS
                : Yolo.DEFAULT_CPU_LOG_EVERY_N_STEPS);
        map.put("backend_log_path", new File(runDir(), "training.log").getAbsolutePath());
        map.put("ui_log_path", new File(getOutputModelDir(), "training-ui.log").getAbsolutePath());
        return map;
    }

    private Map<String, Object> outputsMap() {
        Map<String, Object> map = new LinkedHashMap<String, Object>();
        map.put("output_dir", getOutputModelDir().getAbsolutePath());
        map.put("run_dir", runDir().getAbsolutePath());
        map.put("config_file", getConfigFile().getAbsolutePath());
        map.put("exported_model_file", new File(outputWeightsPath).getAbsoluteFile().getAbsolutePath());
        map.put("best_checkpoint", new File(new File(runDir(), "weights"), "best.pt").getAbsolutePath());
        map.put("last_checkpoint", new File(new File(runDir(), "weights"), "last.pt").getAbsolutePath());
        return map;
    }

    private String modelSource() {
        return fineTune && baseModelPath != null && !baseModelPath.trim().isEmpty()
                ? new File(baseModelPath).getAbsolutePath()
                : YoloModelRegistry.resolveScratchArchitecture(scratchArchitecture);
    }

    private Object trainDevice() {
        return "cuda".equals(device) ? Integer.valueOf(0) : device;
    }

    private String runName() {
        String name = new File(outputWeightsPath).getName();
        return name.toLowerCase().endsWith(YoloModelRegistry.YOLO_WEIGHTS_EXTENSION)
                ? name.substring(0, name.length() - YoloModelRegistry.YOLO_WEIGHTS_EXTENSION.length())
                : name;
    }

    private File projectDir() {
        File outputDir = getOutputModelDir();
        File parent = outputDir.getParentFile();
        return parent != null && outputDir.getName().equals(runName()) ? parent : outputDir;
    }

    private File runDir() {
        return new File(projectDir(), runName()).getAbsoluteFile();
    }

    private static String absolutePath(File file) {
        return file == null ? null : file.getAbsoluteFile().getAbsolutePath();
    }

    private static void flatten(String prefix, Object value, List<String> lines) {
        if (value instanceof Map) {
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
                String key = prefix.isEmpty() ? String.valueOf(entry.getKey())
                        : prefix + "." + String.valueOf(entry.getKey());
                flatten(key, entry.getValue(), lines);
            }
            return;
        }
        lines.add(prefix + "=" + String.valueOf(value));
    }

    /**
     * Creates a YoloTrainingConfig from the ui.
     *
     * @param modelName the model name.
     * @param datasetPath the dataset path.
     * @param epochs the epochs.
     * @param fineTune the fine tune.
     * @param baseModelPath the base model path.
     * @param scratchArchitecture the scratch architecture.
     * @param modelsDir the models directory.
     * @return the created yolo training config.
     */
    public static YoloTrainingConfig fromUi(String modelName, String datasetPath, int epochs,
            boolean fineTune, String baseModelPath, String scratchArchitecture, String modelsDir) {
        return fromUi(modelName, datasetPath, epochs, fineTune, baseModelPath, scratchArchitecture, modelsDir, "cpu");
    }

    /**
     * Creates a YoloTrainingConfig from the ui.
     *
     * @param modelName the model name.
     * @param datasetPath the dataset path.
     * @param epochs the epochs.
     * @param fineTune the fine tune.
     * @param baseModelPath the base model path.
     * @param scratchArchitecture the scratch architecture.
     * @param modelsDir the models directory.
     * @param device the device.
     * @return the created yolo training config.
     */
    public static YoloTrainingConfig fromUi(String modelName, String datasetPath, int epochs,
            boolean fineTune, String baseModelPath, String scratchArchitecture, String modelsDir, String device) {
        String normalizedName = normalizeModelName(modelName);
        File yoloDir = modelsDir == null
                ? new File(YoloModelRegistry.YOLO_MODELS_SUBDIR)
                : new File(modelsDir, YoloModelRegistry.YOLO_MODELS_SUBDIR);
        File outputDir = TrainingModelPaths.uniqueModelDir(yoloDir, normalizedName,
                YoloModelRegistry.YOLO_WEIGHTS_EXTENSION);
        String actualName = outputDir.getName();
        File output = new File(outputDir, actualName + YoloModelRegistry.YOLO_WEIGHTS_EXTENSION);
        return new YoloTrainingConfig(actualName, datasetPath, epochs, DEFAULT_IMAGE_SIZE,
                fineTune, fineTune ? baseModelPath : null, fineTune ? null : scratchArchitecture,
                modelsDir, output.getAbsolutePath(), DEFAULT_PREVIEW_EPOCH_PERIOD, device);
    }

    private static String normalizeModelName(String modelName) {
        if (modelName == null) {
            return "";
        }
        String name = modelName.trim();
        if (name.toLowerCase().endsWith(YoloModelRegistry.YOLO_WEIGHTS_EXTENSION)) {
            name = name.substring(0, name.length() - YoloModelRegistry.YOLO_WEIGHTS_EXTENSION.length());
        }
        return name;
    }

    private static String normalizeDevice(String device) {
        if (device == null) {
            return "cpu";
        }
        String normalized = device.trim().toLowerCase();
        return "cuda".equals(normalized) || "mps".equals(normalized) ? normalized : "cpu";
    }
}
