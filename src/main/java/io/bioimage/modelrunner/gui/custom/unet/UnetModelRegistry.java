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

public final class UnetModelRegistry {

    public static final String UNET_MODELS_SUBDIR = "unet";
    public static final String UNET_WEIGHTS_EXTENSION = ".pt";
    public static final String UNET_PYTORCH_WEIGHTS_EXTENSION = ".pth";

    private static final String[][] SCRATCH_ARCHITECTURES = new String[][] {
            {"Tiny 2D", "tiny-2d"},
            {"Tiny 2.5D", "tiny-2.5d"},
            {"Medium 2D", "medium-2d"},
            {"Medium 2.5D", "medium-2.5d"}
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
        LinkedHashMap<String, String> architectures = new LinkedHashMap<String, String>();
        for (String[] architecture : SCRATCH_ARCHITECTURES) {
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
        for (String[] candidate : SCRATCH_ARCHITECTURES) {
            if (candidate[1].equalsIgnoreCase(architecture.trim())) {
                return true;
            }
        }
        return false;
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
