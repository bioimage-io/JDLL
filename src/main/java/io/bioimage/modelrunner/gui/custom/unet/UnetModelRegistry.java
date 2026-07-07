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
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

import io.bioimage.modelrunner.gui.custom.training.TrainingConfigFiles;

public final class UnetModelRegistry {

    public static final String UNET_MODELS_SUBDIR = "unet";
    public static final String UNET_WEIGHTS_EXTENSION = ".pt";
    public static final String UNET_PYTORCH_WEIGHTS_EXTENSION = ".pth";

    public static final String SMALL_2D = "tiny-2d";
    public static final String MEDIUM_2D = "medium-2d";
    public static final String SMALL_FAST_3D = "tiny-2.5d";
    public static final String MEDIUM_FAST_3D = "medium-2.5d";
    public static final String SMALL_TRUE_3D = "tiny-3d";
    public static final String MEDIUM_TRUE_3D = "medium-3d";

    private static final boolean TRUE_3D_SUPPORTED = true;

    private static final String[][] PLANAR_SCRATCH_ARCHITECTURES = new String[][] {
            {"Small", SMALL_2D},
            {"Medium", MEDIUM_2D}
    };

    private static final String[][] FAST_3D_SCRATCH_ARCHITECTURES = new String[][] {
            {"Small - Fast 3D", SMALL_FAST_3D},
            {"Medium - Fast 3D", MEDIUM_FAST_3D}
    };

    private static final String[][] TRUE_3D_SCRATCH_ARCHITECTURES = new String[][] {
            {"Small - True 3D", SMALL_TRUE_3D},
            {"Medium - True 3D", MEDIUM_TRUE_3D}
    };

    private static final String[] PREFERRED_WEIGHTS = new String[] {
            "model.pt", "model.pth", "weights_best.pt", "weights_best.pth",
            "best.pt", "best.pth", "weights_last.pt", "weights_last.pth",
            "last.pt", "last.pth"
    };

    private UnetModelRegistry() {}

    /**
     * Builds the model entries.
     *
     * @param modelsDir the models directory.
     * @return the created linked hash map.
     */
    public static LinkedHashMap<String, String> buildModelEntries(String modelsDir) {
        LinkedHashMap<String, String> models = new LinkedHashMap<String, String>();
        File unetDir = modelsDir == null ? new File(UNET_MODELS_SUBDIR) : new File(modelsDir, UNET_MODELS_SUBDIR);

        File[] customModelDirs = unetDir.listFiles(file -> file.isDirectory() && isModelDirectory(file));
        if (customModelDirs != null) {
            Arrays.sort(customModelDirs, Comparator.comparing(File::getName, String.CASE_INSENSITIVE_ORDER));
            for (File modelDir : customModelDirs) {
                File modelFile = findModelFile(modelDir);
                models.put("[Custom] " + modelDir.getName(), modelFile.getAbsolutePath());
            }
        }

        File[] customModels = unetDir.listFiles(file -> file.isFile() && isWeightsFile(file.getName()));
        if (customModels == null) {
            return models;
        }
        Arrays.sort(customModels, Comparator.comparing(File::getName, String.CASE_INSENSITIVE_ORDER));
        for (File modelFile : customModels) {
            models.put("[Custom] " + removeWeightsExtension(modelFile.getName()), modelFile.getAbsolutePath());
        }
        return models;
    }

    /**
     * Builds the scratch architecture entries.
     *
     * @return the created linked hash map.
     */
    public static LinkedHashMap<String, String> buildScratchArchitectureEntries() {
        return buildPlanarScratchArchitectureEntries();
    }

    /**
     * Builds the scratch architecture entries for 2D datasets.
     *
     * @return the created linked hash map.
     */
    public static LinkedHashMap<String, String> buildPlanarScratchArchitectureEntries() {
        return buildPlanarScratchArchitectureEntries(null, null, false);
    }

    /**
     * Builds the scratch architecture entries for 2D datasets.
     *
     * @param modelsDir the models directory.
     * @param modelName the model name.
     * @param includeCustom whether a compatible custom config may be added.
     * @return the created linked hash map.
     */
    public static LinkedHashMap<String, String> buildPlanarScratchArchitectureEntries(String modelsDir,
            String modelName, boolean includeCustom) {
        LinkedHashMap<String, String> architectures = new LinkedHashMap<String, String>();
        String customName = normalizeModelName(modelName);
        String customConfig = includeCustom ? customScratchConfigValue(modelsDir, modelName, false) : null;
        if (customConfig != null) {
            architectures.put("[Custom config] " + customName, customConfig);
        }
        addArchitectures(architectures, PLANAR_SCRATCH_ARCHITECTURES);
        return architectures;
    }

    /**
     * Builds the scratch architecture entries for datasets that look volumetric.
     *
     * @return the created linked hash map.
     */
    public static LinkedHashMap<String, String> buildVolumeScratchArchitectureEntries() {
        return buildVolumeScratchArchitectureEntries(null, null, false);
    }

    /**
     * Builds the scratch architecture entries for datasets that look volumetric.
     *
     * @param modelsDir the models directory.
     * @param modelName the model name.
     * @param includeCustom whether a compatible custom config may be added.
     * @return the created linked hash map.
     */
    public static LinkedHashMap<String, String> buildVolumeScratchArchitectureEntries(String modelsDir,
            String modelName, boolean includeCustom) {
        LinkedHashMap<String, String> architectures = new LinkedHashMap<String, String>();
        String customName = normalizeModelName(modelName);
        String customConfig = includeCustom ? customScratchConfigValue(modelsDir, modelName, true) : null;
        if (customConfig != null) {
            architectures.put("[Custom config] " + customName, customConfig);
        }
        addArchitectures(architectures, FAST_3D_SCRATCH_ARCHITECTURES);
        if (TRUE_3D_SUPPORTED) {
            addArchitectures(architectures, TRUE_3D_SCRATCH_ARCHITECTURES);
        }
        return architectures;
    }

    /**
     * Returns the default architecture for the detected dimensionality.
     *
     * @param volume true if dataset looks volumetric.
     * @return the default architecture value.
     */
    public static String defaultScratchArchitecture(boolean volume) {
        return volume ? SMALL_FAST_3D : SMALL_2D;
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
        if (containsArchitecture(PLANAR_SCRATCH_ARCHITECTURES, architecture)
                || containsArchitecture(FAST_3D_SCRATCH_ARCHITECTURES, architecture)) {
            return true;
        }
        if (!TRUE_3D_SUPPORTED) {
            return false;
        }
        return containsArchitecture(TRUE_3D_SCRATCH_ARCHITECTURES, architecture)
                || isCustomScratchConfig(architecture);
    }

    /**
     * Returns a custom scratch config value.
     *
     * @param modelsDir the models directory.
     * @param modelName the model name.
     * @param volume true for volumetric datasets.
     * @return the config path, or null.
     */
    public static String customScratchConfigValue(String modelsDir, String modelName, boolean volume) {
        File config = TrainingConfigFiles.configFileForModelName(modelsDir, UNET_MODELS_SUBDIR,
                normalizeModelName(modelName));
        return isCustomScratchConfigCompatible(config, volume) ? config.getAbsolutePath() : null;
    }

    /**
     * Returns whether value points to a valid UNet scratch config.
     *
     * @param value the value.
     * @return true if valid.
     */
    public static boolean isCustomScratchConfig(String value) {
        return TrainingConfigFiles.isConfigPath(value) && isCustomScratchConfig(new File(value.trim()));
    }

    /**
     * Loads a custom UNet config.
     *
     * @param value the value.
     * @return the loaded config, or null.
     */
    public static Map<String, Object> loadCustomScratchConfig(String value) {
        return isCustomScratchConfig(value) ? TrainingConfigFiles.load(value) : null;
    }

    /**
     * Returns the architecture encoded by a UNet config.
     *
     * @param config the config.
     * @return the architecture, or null.
     */
    public static String architectureFromConfig(Map<String, Object> config) {
        String architecture = TrainingConfigFiles.stringAt(config, "architecture");
        if (architecture == null || architecture.trim().isEmpty()) {
            architecture = TrainingConfigFiles.stringAt(config, "training", "architecture");
        }
        return architecture;
    }

    private static boolean isCustomScratchConfigCompatible(File configFile, boolean volume) {
        Map<String, Object> config = TrainingConfigFiles.load(configFile);
        if (!isCustomScratchConfig(config)) {
            return false;
        }
        String architecture = architectureFromConfig(config);
        String lowerArchitecture = architecture == null ? "" : architecture.toLowerCase(Locale.ROOT);
        boolean architectureVolume = lowerArchitecture.contains("2.5d") || lowerArchitecture.contains("3d");
        return architectureVolume == volume;
    }

    private static boolean isCustomScratchConfig(File configFile) {
        return isCustomScratchConfig(TrainingConfigFiles.load(configFile));
    }

    private static boolean isCustomScratchConfig(Map<String, Object> config) {
        if (config == null) {
            return false;
        }
        String format = TrainingConfigFiles.stringAt(config, "format");
        String framework = TrainingConfigFiles.stringAt(config, "framework");
        if (format != null && !"jdll-unet".equalsIgnoreCase(format)) {
            return false;
        }
        if (framework != null && !"unet".equalsIgnoreCase(framework)) {
            return false;
        }
        return isKnownBuiltInArchitecture(architectureFromConfig(config));
    }

    private static void addArchitectures(LinkedHashMap<String, String> target, String[][] architectures) {
        for (String[] architecture : architectures) {
            target.put(architecture[0], architecture[1]);
        }
    }

    private static boolean containsArchitecture(String[][] architectures, String architecture) {
        if (architecture == null) {
            return false;
        }
        for (String[] candidate : architectures) {
            if (candidate[1].equalsIgnoreCase(architecture.trim())) {
                return true;
            }
        }
        return false;
    }

    private static boolean isKnownBuiltInArchitecture(String architecture) {
        return containsArchitecture(PLANAR_SCRATCH_ARCHITECTURES, architecture)
                || containsArchitecture(FAST_3D_SCRATCH_ARCHITECTURES, architecture)
                || (TRUE_3D_SUPPORTED && containsArchitecture(TRUE_3D_SCRATCH_ARCHITECTURES, architecture));
    }

    /**
     * Returns whether model path.
     *
     * @param path the path.
     * @return true if model path; false otherwise.
     */
    public static boolean isModelPath(String path) {
        if (path == null || path.trim().isEmpty()) {
            return false;
        }
        File file = new File(path.trim());
        return file.isDirectory() ? isModelDirectory(file) : file.isFile() && isWeightsFile(file.getName());
    }

    /**
     * Returns the model file in a model directory.
     *
     * @param dir the directory.
     * @return the model file, or null.
     */
    public static File findModelFile(File dir) {
        if (dir == null || !dir.isDirectory()) {
            return null;
        }
        File namedWeights = new File(dir, dir.getName() + UNET_WEIGHTS_EXTENSION);
        if (namedWeights.isFile()) {
            return namedWeights;
        }
        namedWeights = new File(dir, dir.getName() + UNET_PYTORCH_WEIGHTS_EXTENSION);
        if (namedWeights.isFile()) {
            return namedWeights;
        }
        for (String name : PREFERRED_WEIGHTS) {
            File candidate = new File(dir, name);
            if (candidate.isFile()) {
                return candidate;
            }
        }
        File[] weights = dir.listFiles(file -> file.isFile() && isWeightsFile(file.getName()));
        if (weights == null || weights.length == 0) {
            return null;
        }
        Arrays.sort(weights, Comparator.comparing(File::getName, String.CASE_INSENSITIVE_ORDER));
        return weights[0];
    }

    /**
     * Removes the weights extension.
     *
     * @param fileName the file name.
     * @return the file name without extension.
     */
    public static String removeWeightsExtension(String fileName) {
        if (fileName == null) {
            return "";
        }
        String lower = fileName.toLowerCase(Locale.ROOT);
        if (lower.endsWith(UNET_WEIGHTS_EXTENSION)) {
            return fileName.substring(0, fileName.length() - UNET_WEIGHTS_EXTENSION.length());
        }
        if (lower.endsWith(UNET_PYTORCH_WEIGHTS_EXTENSION)) {
            return fileName.substring(0, fileName.length() - UNET_PYTORCH_WEIGHTS_EXTENSION.length());
        }
        return fileName;
    }

    private static String normalizeModelName(String modelName) {
        return removeWeightsExtension(modelName == null ? "" : modelName.trim()).trim();
    }

    private static boolean isModelDirectory(File file) {
        return findModelFile(file) != null;
    }

    private static boolean isWeightsFile(String fileName) {
        if (fileName == null) {
            return false;
        }
        String lower = fileName.toLowerCase(Locale.ROOT);
        return lower.endsWith(UNET_WEIGHTS_EXTENSION) || lower.endsWith(UNET_PYTORCH_WEIGHTS_EXTENSION);
    }
}
