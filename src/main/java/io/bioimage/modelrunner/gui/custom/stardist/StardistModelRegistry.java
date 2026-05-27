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
import java.util.Optional;
import java.util.stream.Stream;

public final class StardistModelRegistry {

    public static final String STARDIST_MODELS_SUBDIR = "stardist";
    public static final String STARDIST_WEIGHTS_EXTENSION = ".mpk";
    public static final String STARDIST_ARCHITECTURE_EXTENSION = ".json";
    public static final String PRETRAINED_URL_FORMAT = "https://github.com/ultralytics/assets/releases/download/v8.4.0/%s";

    private static final String[][] PRETRAINED_MODELS = new String[][] {
        {"gray_small", "gray_small.mpk"},
        {"gray_medium", "gray_medium.mpk"},
        {"gray_big", "gray_big.mpk"},
        {"color_small", "color_small.mpk"},
        {"color_medium", "color_medium.mpk"},
        {"color_big", "color_big.mpk"},
    };
    private static final String[][] SCRATCH_ARCHITECTURES = new String[][] {
        {"gray_small", "gray_small.json"},
        {"gray_medium", "gray_medium.json"},
        {"gray_big", "gray_big.json"},
        {"color_small", "color_small.json"},
        {"color_medium", "color_medium.json"},
        {"color_big", "color_big.json"},
    };

    private static final Map<String, Long> PRETRAINED_WEIGHTS_SIZE;
    static {
        PRETRAINED_WEIGHTS_SIZE = new HashMap<String, Long>();
        PRETRAINED_WEIGHTS_SIZE.put("gray_small.mpk", 5_544_453L);
        PRETRAINED_WEIGHTS_SIZE.put("gray_medium.mpk", 5_544_453L);
        PRETRAINED_WEIGHTS_SIZE.put("gray_big.mpk", 5_544_453L);
        PRETRAINED_WEIGHTS_SIZE.put("color_small.mpk", 5_544_453L);
        PRETRAINED_WEIGHTS_SIZE.put("color_medium.mpk", 5_544_453L);
        PRETRAINED_WEIGHTS_SIZE.put("color_big.mpk", 5_544_453L);
    }

    private StardistModelRegistry() {}

    public static LinkedHashMap<String, String> buildModelEntries(String modelsDir) {
        LinkedHashMap<String, String> models = new LinkedHashMap<String, String>();
        File stardistDir = modelsDir == null ? new File(STARDIST_MODELS_SUBDIR) : new File(modelsDir, STARDIST_MODELS_SUBDIR);

        for (String[] pretrained : PRETRAINED_MODELS) {
            models.put("[Pretrained] " + pretrained[0], new File(stardistDir, pretrained[1]).getAbsolutePath());
        }

        File[] customModels = stardistDir.listFiles(file ->
                (file.isDirectory() && isModelDirectory(file))
                || (file.isFile()
                        && file.getName().toLowerCase().endsWith(STARDIST_WEIGHTS_EXTENSION)
                        && !isPretrainedWeightsFile(file.getName())));
        if (customModels == null) {
            return models;
        }
        Arrays.sort(customModels, Comparator.comparing(File::getName, String.CASE_INSENSITIVE_ORDER));
        for (File modelFile : customModels) {
            models.put("[Custom] " + removeWeightsExtension(modelFile.getName()), findMpk(modelFile.getAbsolutePath()).toString());
        }
        return models;
    }

    public static LinkedHashMap<String, String> buildScratchArchitectureEntries() {
        LinkedHashMap<String, String> architectures = new LinkedHashMap<String, String>();
        for (String[] architecture : SCRATCH_ARCHITECTURES) {
            architectures.put(architecture[0], architecture[1]);
        }
        return architectures;
    }

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

    public static boolean isPretrainedWeightsFile(String fileName) {
        return expectedPretrainedSize(fileName) != null;
    }

    public static Long expectedPretrainedSize(String fileName) {
        return fileName == null ? null : PRETRAINED_WEIGHTS_SIZE.get(fileName.toLowerCase());
    }

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
        Long expectedSize = expectedPretrainedSize(modelFile.getName());
        return expectedSize == null || expectedSize.longValue() == modelFile.length();
    }

    public static boolean canDownload(String modelPath) {
        return modelPath != null && expectedPretrainedSize(new File(modelPath).getName()) != null;
    }

    public static String downloadUrl(String modelPath) {
        return String.format(PRETRAINED_URL_FORMAT, new File(modelPath).getName());
    }

    private static String removeWeightsExtension(String fileName) {
        if (fileName.toLowerCase().endsWith(STARDIST_WEIGHTS_EXTENSION)) {
            return fileName.substring(0, fileName.length() - STARDIST_WEIGHTS_EXTENSION.length());
        }
        return fileName;
    }

    private static boolean isModelDirectory(File file) {
        return new File(file, "config.json").isFile() && findMpk(file.getAbsolutePath()) != null;
    }
    
    public static Path findMpk(String dir) {
        try (Stream<Path> files = Files.list(Paths.get(dir))) {
            return files
                    .filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().matches("(?i).*\\.mpk$"))
                    .findFirst().orElse(null);
        } catch (IOException e) {
			return null;
		}
    }
}
