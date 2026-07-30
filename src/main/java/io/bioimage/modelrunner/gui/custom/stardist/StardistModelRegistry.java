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
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Stream;

import io.bioimage.modelrunner.gui.custom.training.TrainingConfigFiles;

public final class StardistModelRegistry {

    public static final String STARDIST_MODELS_SUBDIR = "stardist";
    public static final String STARDIST_KERAS_WEIGHTS_EXTENSION = ".h5";
    public static final String STARDIST_ARCHITECTURE_EXTENSION = ".json";
    public static final String PRETRAINED_URL_FORMAT = "https://github.com/stardist/stardist-models/releases/download/v0.1/python_%s.zip";

    private static final String[][] PRETRAINED_MODELS = new String[][] {
        {"2D_versatile_fluo", "2D_versatile_fluo" + File.separator + "weights_best.h5"},
        {"2D_versatile_he", "2D_versatile_he" + File.separator + "weights_best.h5"},
    };
    private static final String[][] SCRATCH_ARCHITECTURES_2D = new String[][] {
        {"small", "gray_small.json"},
        {"medium", "gray_medium.json"},
        {"medium-big", "gray_medium_big.json"},
        {"big", "gray_big.json"},
        {"large", "gray_large.json"},
    };
    private static final String[][] SCRATCH_ARCHITECTURES_3D = new String[][] {
        {"small 3D", "gray_3d_small.json"},
        {"medium 3D", "gray_3d_medium.json"},
        {"medium-big 3D", "gray_3d_medium_big.json"},
        {"big 3D", "gray_3d_big.json"},
        {"large 3D", "gray_3d_large.json"},
    };

    private static final Map<String, Long> PRETRAINED_WEIGHTS_SIZE;
    static {
        PRETRAINED_WEIGHTS_SIZE = new HashMap<String, Long>();
        PRETRAINED_WEIGHTS_SIZE.put("2D_versatile_fluo" + File.separator + "weights_best.h5", 5771480L);
        PRETRAINED_WEIGHTS_SIZE.put("2D_versatile_he" + File.separator + "weights_best.h5", 5774704L);
    }

    private StardistModelRegistry() {}

    /**
     * Builds the model entries.
     *
     * @param modelsDir the models directory.
     * @return the created linked hash map.
     */
    public static LinkedHashMap<String, String> buildModelEntries(String modelsDir) {
        LinkedHashMap<String, String> models = new LinkedHashMap<String, String>();
        File stardistDir = modelsDir == null ? new File(STARDIST_MODELS_SUBDIR) : new File(modelsDir, STARDIST_MODELS_SUBDIR);

        for (String[] pretrained : PRETRAINED_MODELS) {
            models.put("[Pretrained] " + pretrained[0], new File(stardistDir, pretrained[1]).getAbsolutePath());
        }

        File[] customModels = stardistDir.listFiles(file ->
                (file.isDirectory() && isModelDirectory(file) && !models.containsValue(findModelFile(file.getAbsolutePath()).toString())));
        if (customModels == null) {
            return models;
        }
        Arrays.sort(customModels, Comparator.comparing(File::getName, String.CASE_INSENSITIVE_ORDER));
        for (File modelFile : customModels) {
            models.put("[Custom] " + removeWeightsExtension(modelFile.getName()), findModelFile(modelFile.getAbsolutePath()).toString());
        }
        return models;
    }

    /**
     * Builds the scratch architecture entries.
     *
     * @return the created linked hash map.
     */
    public static LinkedHashMap<String, String> buildScratchArchitectureEntries() {
        return buildScratchArchitectureEntries(null, null, false, false);
    }

    /**
     * Builds the scratch architecture entries, adding a compatible custom config if present.
     *
     * @param modelsDir the models directory.
     * @param modelName the model name.
     * @return the created linked hash map.
     */
    public static LinkedHashMap<String, String> buildScratchArchitectureEntries(String modelsDir, String modelName) {
        return buildScratchArchitectureEntries(modelsDir, modelName, false, true);
    }

    public static LinkedHashMap<String, String> buildScratchArchitectureEntries(String modelsDir, String modelName,
            boolean volume, boolean datasetReviewed) {
        LinkedHashMap<String, String> architectures = new LinkedHashMap<String, String>();
        String customName = normalizeModelName(modelName);
        String customConfig = datasetReviewed ? customScratchConfigValue(modelsDir, modelName, volume) : null;
        if (customConfig != null) {
            architectures.put("[Custom config] " + customName, customConfig);
        }
        for (String[] architecture : volume ? SCRATCH_ARCHITECTURES_3D : SCRATCH_ARCHITECTURES_2D) {
            architectures.put(architecture[0], architecture[1]);
        }
        return architectures;
    }

    /**
     * Returns whether known scratch architecture.
     *
     * @param architecture the architecture.
     * @return true if known scratch architecture; false otherwise.
     */
    public static boolean isKnownScratchArchitecture(String architecture) {
        if (architecture == null) {
            return false;
        }
        for (String[][] architectures : new String[][][] {SCRATCH_ARCHITECTURES_2D, SCRATCH_ARCHITECTURES_3D}) {
            for (String[] candidate : architectures) {
                if (candidate[1].equalsIgnoreCase(architecture.trim())) {
                    return true;
                }
            }
        }
        return isCustomScratchConfig(architecture);
    }

    public static boolean isArchitecture3D(String architecture) {
        if (architecture == null) {
            return false;
        }
        for (String[] candidate : SCRATCH_ARCHITECTURES_3D) {
            if (candidate[1].equalsIgnoreCase(architecture.trim())) {
                return true;
            }
        }
        Map<String, Object> custom = loadCustomScratchConfig(architecture);
        return custom != null && configDimensions(custom) == 3;
    }

    /**
     * Returns the custom scratch config value for a model name.
     *
     * @param modelsDir the models directory.
     * @param modelName the model name.
     * @return the config path, or null.
     */
    public static String customScratchConfigValue(String modelsDir, String modelName) {
        return customScratchConfigValue(modelsDir, modelName, false);
    }

    public static String customScratchConfigValue(String modelsDir, String modelName, boolean volume) {
        File config = TrainingConfigFiles.configFileForModelName(modelsDir, STARDIST_MODELS_SUBDIR,
                normalizeModelName(modelName));
        return isCustomScratchConfig(config, volume) ? config.getAbsolutePath() : null;
    }

    /**
     * Returns whether value points to a valid StarDist scratch config.
     *
     * @param value the value.
     * @return true if valid.
     */
    public static boolean isCustomScratchConfig(String value) {
        return TrainingConfigFiles.isConfigPath(value) && isCustomScratchConfig(new File(value.trim()));
    }

    /**
     * Loads a custom StarDist config.
     *
     * @param value the config path.
     * @return the loaded config, or null.
     */
    public static Map<String, Object> loadCustomScratchConfig(String value) {
        return isCustomScratchConfig(value) ? TrainingConfigFiles.load(value) : null;
    }

    /**
     * Returns the image channel mode for a scratch architecture.
     *
     * @param architecture the architecture or custom config path.
     * @return "rgb" or "grayscale".
     */
    public static String imageChannelsForScratchArchitecture(String architecture) {
        Map<String, Object> custom = loadCustomScratchConfig(architecture);
        if (custom != null) {
            Object channels = custom.get("n_channel_in");
            if (channels != null && "3".equals(channels.toString())) {
                return "rgb";
            }
        }
        String arch = architecture == null ? "" : architecture.toLowerCase();
        return arch.startsWith("color") ? "rgb" : "grayscale";
    }

    private static boolean isCustomScratchConfig(File configFile) {
        return isCustomScratchConfig(configFile, null);
    }

    private static boolean isCustomScratchConfig(File configFile, Boolean volume) {
        Map<String, Object> config = TrainingConfigFiles.load(configFile);
        if (config == null) {
            return false;
        }
        String framework = TrainingConfigFiles.stringAt(config, "framework");
        if (framework != null && !"stardist".equalsIgnoreCase(framework)) {
            return false;
        }
        if (volume != null && (configDimensions(config) == 3) != volume.booleanValue()) {
            return false;
        }
        return config.containsKey("n_rays") || config.containsKey("grid") || config.containsKey("backbone")
                || config.containsKey("train_patch_size") || config.containsKey("train_batch_size");
    }

    /**
     * Returns whether pretrained weights file.
     *
     * @param fileName the file name.
     * @return true if pretrained weights file; false otherwise.
     */
    public static boolean isPretrainedWeightsFile(String fileName) {
        return expectedPretrainedSize(fileName) != null;
    }

    /**
     * Returns the result of expected pretrained size.
     *
     * @param fileName the file name.
     * @return the resulting long.
     */
    public static Long expectedPretrainedSize(String fileName) {
        return fileName == null ? null : PRETRAINED_WEIGHTS_SIZE.get(fileName);
    }

    /**
     * Returns whether installed.
     *
     * @param modelPath the model path.
     * @return true if installed; false otherwise.
     */
    public static boolean isInstalled(String modelPath) {
        if (modelPath == null) {
            return false;
        }
        File modelFile = new File(modelPath);
        if (modelFile.isDirectory()) {
            return isModelDirectory(modelFile);
        }
        if (!modelFile.isFile()) {
            return false;
        }
        String modelAndFile = modelFile.getParentFile().getName() + File.separator + modelFile.getName();
        Long expectedSize = expectedPretrainedSize(modelAndFile);
        return expectedSize == null || expectedSize.longValue() == modelFile.length();
    }

    /**
     * Returns whether can download.
     *
     * @param modelPath the model path.
     * @return true if can download; false otherwise.
     */
    public static boolean canDownload(String modelPath) {
        if (modelPath == null) {
            return false;
        }
        File modelFile = new File(modelPath);
        File parent = modelFile.getParentFile();
        if (parent == null) {
            return false;
        }
        return expectedPretrainedSize(parent.getName() + File.separator + modelFile.getName()) != null;
    }

    /**
     * Returns the result of download URL.
     *
     * @param modelPath the model path.
     * @return the resulting string.
     */
    public static String downloadUrl(String modelPath) {
        return String.format(PRETRAINED_URL_FORMAT, new File(modelPath).getName());
    }

    private static String removeWeightsExtension(String fileName) {
        if (fileName == null) {
            return "";
        }
        if (fileName.toLowerCase().endsWith(STARDIST_KERAS_WEIGHTS_EXTENSION)) {
            return fileName.substring(0, fileName.length() - STARDIST_KERAS_WEIGHTS_EXTENSION.length());
        }
        return fileName;
    }

    private static String normalizeModelName(String modelName) {
        return removeWeightsExtension(modelName == null ? "" : modelName.trim()).trim();
    }

    private static boolean isModelDirectory(File file) {
        return new File(file, "config.json").isFile() && findModelFile(file.getAbsolutePath()) != null;
    }
    
    /**
     * Returns the result of find mpk.
     *
     * @param dir the directory.
     * @return the resulting path.
     */
    public static Path findMpk(String dir) {
        return findModelFile(dir);
    }

    /**
     * Returns the result of find model file.
     *
     * @param dir the directory.
     * @return the resulting path.
     */
    public static Path findModelFile(String dir) {
        try (Stream<Path> files = Files.list(Paths.get(dir))) {
            return files
                    .filter(Files::isRegularFile)
                    .filter(p -> isWeightsFile(p.getFileName().toString()))
                    .sorted(Comparator.comparingInt(StardistModelRegistry::weightsPriority)
                            .thenComparing(p -> p.getFileName().toString(), String.CASE_INSENSITIVE_ORDER))
                    .findFirst().orElse(null);
        } catch (IOException e) {
			return null;
		}
    }

    /**
     * Resolves a model that can be used as a StarDist fine-tuning source.
     *
     * @param modelPath a model directory or HDF5 weights file.
     * @return the resolved source, or {@code null} when it is incomplete or incompatible.
     */
    public static FineTuneSource resolveFineTuneSource(String modelPath) {
        if (modelPath == null || modelPath.trim().isEmpty()) {
            return null;
        }
        File selected = new File(modelPath.trim()).getAbsoluteFile();
        File modelDir = selected.isDirectory() ? selected : selected.getParentFile();
        if (modelDir == null) {
            return null;
        }
        Path weights = selected.isFile() && isWeightsFile(selected.getName())
                ? selected.toPath() : findModelFile(modelDir.getAbsolutePath());
        File configFile = new File(modelDir, TrainingConfigFiles.CONFIG_FILE_NAME);
        Map<String, Object> config = TrainingConfigFiles.load(configFile);
        if (weights == null || !Files.isRegularFile(weights) || !isCompatibleConfig(config, null)) {
            return null;
        }
        return new FineTuneSource(modelDir, weights.toFile(), configFile, config);
    }

    /**
     * Returns whether a path is an installed or downloadable fine-tuning source.
     *
     * @param modelPath the model path.
     * @return true when the source can be selected.
     */
    public static boolean isSelectableFineTuneSource(String modelPath) {
        return resolveFineTuneSource(modelPath) != null || canDownload(modelPath);
    }

    public static boolean isSelectableFineTuneSource(String modelPath, int dimensions) {
        FineTuneSource source = resolveFineTuneSource(modelPath);
        if (source != null) {
            return isCompatibleConfig(source.getConfig(), Integer.valueOf(dimensions));
        }
        return dimensions == 2 && canDownload(modelPath);
    }

    public static int configDimensions(Map<String, Object> config) {
        if (config == null) {
            return 2;
        }
        Object dimensions = config.get("n_dim");
        if (dimensions != null && "3".equals(dimensions.toString())) {
            return 3;
        }
        String axes = TrainingConfigFiles.stringAt(config, "axes");
        return axes != null && axes.toUpperCase().contains("Z") ? 3 : 2;
    }

    private static boolean isCompatibleConfig(Map<String, Object> config, Integer dimensions) {
        if (config == null) {
            return false;
        }
        String framework = TrainingConfigFiles.stringAt(config, "framework");
        if (framework != null && !"stardist".equalsIgnoreCase(framework)) {
            return false;
        }
        return (dimensions == null || configDimensions(config) == dimensions.intValue())
                && config.containsKey("n_channel_in")
                && config.containsKey("n_rays")
                && config.containsKey("grid")
                && config.containsKey("backbone");
    }

    private static int weightsPriority(Path path) {
        String name = path.getFileName().toString().toLowerCase();
        if ("weights_best.h5".equals(name)) {
            return 0;
        }
        if ("weights_last.h5".equals(name)) {
            return 1;
        }
        if (name.endsWith(STARDIST_KERAS_WEIGHTS_EXTENSION)) {
            return 2;
        }
        return 3;
    }

    private static boolean isWeightsFile(String fileName) {
        if (fileName == null) {
            return false;
        }
        String lower = fileName.toLowerCase();
        return lower.endsWith(STARDIST_KERAS_WEIGHTS_EXTENSION);
    }

    /**
     * Resolved immutable StarDist fine-tuning source.
     */
    public static final class FineTuneSource {
        private final File modelDirectory;
        private final File weightsFile;
        private final File configFile;
        private final Map<String, Object> config;

        private FineTuneSource(File modelDirectory, File weightsFile, File configFile,
                Map<String, Object> config) {
            this.modelDirectory = modelDirectory;
            this.weightsFile = weightsFile;
            this.configFile = configFile;
            this.config = new LinkedHashMap<String, Object>(config);
        }

        public File getModelDirectory() {
            return modelDirectory;
        }

        public File getWeightsFile() {
            return weightsFile;
        }

        public File getConfigFile() {
            return configFile;
        }

        public Map<String, Object> getConfig() {
            return new LinkedHashMap<String, Object>(config);
        }
    }
}
