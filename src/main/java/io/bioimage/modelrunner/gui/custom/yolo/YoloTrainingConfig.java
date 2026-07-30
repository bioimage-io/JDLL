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
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import io.bioimage.modelrunner.gui.custom.training.TrainingConfigFiles;
import io.bioimage.modelrunner.gui.custom.training.TrainingModelPaths;
import io.bioimage.modelrunner.model.special.yolo.Yolo;
import io.bioimage.modelrunner.model.special.yolo.YoloTrainingAttemptResult;
import io.bioimage.modelrunner.utils.Constants;

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
    private final Map<String, Object> trainingOptions;
    private final Object requestedBatch;
    private final boolean oomRetry;
    private final int minimumBatch;
    private String resolvedDevice;
    private Integer resolvedBatch;
    private String pythonVersion;
    private String ultralyticsVersion;
    private String torchVersion;
    private String torchvisionVersion;

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
        Map<String, Object> custom = fineTune ? null : TrainingConfigFiles.load(scratchArchitecture);
        this.modelName = modelName;
        this.datasetYamlPath = datasetYamlPath;
        this.epochs = epochs;
        this.imageSize = positiveInt(TrainingConfigFiles.objectAt(custom, "training", "imgsz"), imageSize);
        this.fineTune = fineTune;
        this.baseModelPath = baseModelPath;
        this.scratchArchitecture = scratchArchitecture;
        this.modelsDir = modelsDir;
        this.outputWeightsPath = outputWeightsPath;
        this.previewEpochPeriod = previewEpochPeriod;
        this.device = normalizeDevice(device);
        this.trainingOptions = buildTrainingOptions(custom, epochs, this.imageSize);
        this.requestedBatch = customBatch(custom);
        this.oomRetry = booleanValue(TrainingConfigFiles.objectAt(custom, "runtime", "oom_retry"), true);
        this.minimumBatch = positiveInt(TrainingConfigFiles.objectAt(custom, "runtime", "minimum_batch"), 1);
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
     * Returns the Ultralytics options controlled by the training recipe.
     *
     * @return a defensive option copy.
     */
    public Map<String, Object> getBackendTrainingOptions() {
        Map<String, Object> options = new LinkedHashMap<String, Object>();
        for (Map.Entry<String, Object> entry : trainingOptions.entrySet()) {
            if ("augmentations".equals(entry.getKey()) && entry.getValue() instanceof Map) {
                options.putAll(copyMap(entry.getValue()));
            } else if (!"epochs".equals(entry.getKey()) && !"imgsz".equals(entry.getKey())) {
                options.put(entry.getKey(), entry.getValue());
            }
        }
        return options;
    }

    /**
     * Returns the requested physical batch or {@code "auto"}.
     *
     * @return the batch request.
     */
    public Object getRequestedBatch() {
        return requestedBatch;
    }

    public boolean isOomRetryEnabled() {
        return oomRetry;
    }

    public int getMinimumBatch() {
        return minimumBatch;
    }

    /**
     * Updates runtime values resolved by Python and persists them on the next write.
     *
     * @param result the attempt result.
     */
    public synchronized void updateRuntime(YoloTrainingAttemptResult result) {
        if (result == null) {
            return;
        }
        resolvedDevice = result.getResolvedDevice();
        resolvedBatch = result.getResolvedBatch();
        pythonVersion = result.getPythonVersion();
        ultralyticsVersion = result.getUltralyticsVersion();
        torchVersion = result.getTorchVersion();
        torchvisionVersion = result.getTorchvisionVersion();
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

    public File getTrainingResultsFile() {
        return new File(getOutputModelDir(), "training_results.json");
    }

    /**
     * Writes a JSON training config file.
     *
     * @param resolvedDatasetYaml the actual data.yaml used by Ultralytics.
     * @return the written config file.
     * @throws IOException if an I/O error occurs.
     */
    public synchronized File writeConfig(File resolvedDatasetYaml) throws IOException {
        File configFile = getConfigFile();
        File parent = configFile.getParentFile();
        if (parent != null) {
            Files.createDirectories(parent.toPath());
        }
        File temporary = new File(parent, configFile.getName() + ".tmp");
        try (Writer writer = Files.newBufferedWriter(temporary.toPath(), StandardCharsets.UTF_8)) {
            CONFIG_GSON.toJson(toConfigMap(resolvedDatasetYaml), writer);
        }
        replaceAtomically(temporary, configFile);
        return configFile;
    }

    /**
     * Formats all training parameters for the UI log.
     *
     * @param resolvedDatasetYaml the actual data.yaml used by Ultralytics.
     * @return the formatted parameter lines.
     */
    public synchronized List<String> toLogLines(File resolvedDatasetYaml) {
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
    public synchronized Map<String, Object> toConfigMap(File resolvedDatasetYaml) {
        Map<String, Object> root = new LinkedHashMap<String, Object>();
        root.put("framework", "yolo");
        root.put("format_version", 0);
        root.put("software", softwareMap());
        root.put("model", modelMap());
        root.put("dataset", datasetMap(resolvedDatasetYaml));
        root.put("training", new LinkedHashMap<String, Object>(trainingOptions));
        root.put("runtime", runtimeMap());
        return root;
    }

    private Map<String, Object> softwareMap() {
        Map<String, Object> map = new LinkedHashMap<String, Object>();
        map.put("jdll", Constants.JDLL_VERSION);
        map.put("deepicy", System.getProperty("jdll.deepicy.version", "unknown"));
        map.put("python", pythonVersion);
        map.put("ultralytics", ultralyticsVersion);
        map.put("torch", torchVersion);
        map.put("torchvision", torchvisionVersion);
        return map;
    }

    private Map<String, Object> modelMap() {
        Map<String, Object> map = new LinkedHashMap<String, Object>();
        map.put("name", modelName);
        map.put("task", "detect");
        map.put("start_mode", fineTune ? "fine_tune" : "from_scratch");
        map.put("source", fineTune ? null : modelSource());
        map.put("base_model", fineTune ? absolutePath(new File(baseModelPath)) : null);
        return map;
    }

    private Map<String, Object> datasetMap(File resolvedDatasetYaml) {
        Map<String, Object> map = new LinkedHashMap<String, Object>();
        map.put("requested_path", datasetYamlPath);
        map.put("resolved_yaml", absolutePath(resolvedDatasetYaml));
        return map;
    }

    private synchronized Map<String, Object> runtimeMap() {
        Map<String, Object> map = new LinkedHashMap<String, Object>();
        map.put("requested_device", device);
        map.put("resolved_device", resolvedDevice);
        map.put("requested_batch", requestedBatch);
        map.put("resolved_batch", resolvedBatch);
        map.put("oom_retry", Boolean.valueOf(oomRetry));
        map.put("minimum_batch", Integer.valueOf(minimumBatch));
        map.put("workers", Yolo.DEFAULT_TRAIN_WORKERS);
        return map;
    }

    private String modelSource() {
        return fineTune && baseModelPath != null && !baseModelPath.trim().isEmpty()
                ? new File(baseModelPath).getAbsolutePath()
                : YoloModelRegistry.resolveScratchArchitecture(scratchArchitecture);
    }


    private static String absolutePath(File file) {
        return file == null ? null : file.getAbsoluteFile().getAbsolutePath();
    }

    private static void replaceAtomically(File source, File target) throws IOException {
        try {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
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

    private static Map<String, Object> buildTrainingOptions(Map<String, Object> custom,
            int epochs, int imageSize) {
        Map<String, Object> options = defaultTrainingOptions();
        Map<String, Object> customTraining = TrainingConfigFiles.mapAt(custom, "training");
        if (customTraining == null) {
            customTraining = TrainingConfigFiles.mapAt(custom, "ultralytics_train");
        }
        merge(options, customTraining);
        options.put("epochs", Integer.valueOf(epochs));
        options.put("imgsz", Integer.valueOf(imageSize));
        options.remove("batch");
        for (String key : new String[] {
                "data", "model", "project", "name", "device", "workers", "exist_ok",
                "verbose", "plots", "resume", "save", "save_period"
        }) {
            options.remove(key);
        }
        return Collections.unmodifiableMap(options);
    }

    private static Map<String, Object> defaultTrainingOptions() {
        Map<String, Object> options = new LinkedHashMap<String, Object>();
        options.put("epochs", Integer.valueOf(100));
        options.put("imgsz", Integer.valueOf(DEFAULT_IMAGE_SIZE));
        options.put("seed", Integer.valueOf(Yolo.DEFAULT_TRAIN_SEED));
        options.put("deterministic", Boolean.valueOf(Yolo.DEFAULT_TRAIN_DETERMINISTIC));
        options.put("optimizer", "auto");
        options.put("patience", Integer.valueOf(100));
        options.put("lr0", Double.valueOf(0.01));
        options.put("lrf", Double.valueOf(0.01));
        options.put("momentum", Double.valueOf(0.937));
        options.put("weight_decay", Double.valueOf(0.0005));
        options.put("warmup_epochs", Double.valueOf(3.0));
        options.put("warmup_momentum", Double.valueOf(0.8));
        options.put("warmup_bias_lr", Double.valueOf(0.1));
        options.put("nbs", Integer.valueOf(Yolo.DEFAULT_NOMINAL_BATCH_SIZE));
        options.put("box", Double.valueOf(7.5));
        options.put("cls", Double.valueOf(0.5));
        options.put("cls_pw", Double.valueOf(0.0));
        options.put("dfl", Double.valueOf(1.5));
        options.put("amp", Boolean.TRUE);
        options.put("cos_lr", Boolean.FALSE);
        options.put("close_mosaic", Integer.valueOf(10));
        options.put("cache", Boolean.FALSE);
        options.put("rect", Boolean.FALSE);
        options.put("multi_scale", Double.valueOf(0.0));
        options.put("compile", Boolean.FALSE);
        options.put("freeze", null);
        options.put("fraction", Double.valueOf(1.0));
        options.put("single_cls", Boolean.FALSE);
        options.put("classes", null);
        options.put("val", Boolean.TRUE);
        options.put("split", "val");
        options.put("conf", null);
        options.put("iou", Double.valueOf(0.7));
        options.put("max_det", Integer.valueOf(300));
        Map<String, Object> augmentations = new LinkedHashMap<String, Object>();
        augmentations.put("hsv_h", Double.valueOf(0.015));
        augmentations.put("hsv_s", Double.valueOf(0.7));
        augmentations.put("hsv_v", Double.valueOf(0.4));
        augmentations.put("degrees", Double.valueOf(0.0));
        augmentations.put("translate", Double.valueOf(0.1));
        augmentations.put("scale", Double.valueOf(0.5));
        augmentations.put("shear", Double.valueOf(0.0));
        augmentations.put("perspective", Double.valueOf(0.0));
        augmentations.put("flipud", Double.valueOf(0.0));
        augmentations.put("fliplr", Double.valueOf(0.5));
        augmentations.put("bgr", Double.valueOf(0.0));
        augmentations.put("mosaic", Double.valueOf(1.0));
        augmentations.put("mixup", Double.valueOf(0.0));
        augmentations.put("cutmix", Double.valueOf(0.0));
        augmentations.put("copy_paste", Double.valueOf(0.0));
        options.put("augmentations", augmentations);
        return options;
    }

    private static Object customBatch(Map<String, Object> custom) {
        Object batch = TrainingConfigFiles.objectAt(custom, "runtime", "requested_batch");
        if (batch == null) {
            batch = TrainingConfigFiles.objectAt(custom, "training", "batch");
        }
        if (batch instanceof Number && ((Number) batch).intValue() > 0) {
            return Integer.valueOf(((Number) batch).intValue());
        }
        return "auto";
    }

    private static int positiveInt(Object value, int fallback) {
        if (value instanceof Number && ((Number) value).intValue() > 0) {
            return ((Number) value).intValue();
        }
        return fallback;
    }

    private static boolean booleanValue(Object value, boolean fallback) {
        return value instanceof Boolean ? ((Boolean) value).booleanValue() : fallback;
    }

    @SuppressWarnings("unchecked")
    private static void merge(Map<String, Object> target, Map<String, Object> values) {
        if (values == null) {
            return;
        }
        for (Map.Entry<String, Object> entry : values.entrySet()) {
            Object current = target.get(entry.getKey());
            if (current instanceof Map && entry.getValue() instanceof Map) {
                merge((Map<String, Object>) current, (Map<String, Object>) entry.getValue());
            } else {
                target.put(entry.getKey(), entry.getValue());
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> copyMap(Object value) {
        return value instanceof Map
                ? new LinkedHashMap<String, Object>((Map<String, Object>) value)
                : Collections.<String, Object>emptyMap();
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
