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
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

public final class YoloModelRegistry {

    public static final String YOLO_MODELS_SUBDIR = "yolo";
    public static final String YOLO_WEIGHTS_EXTENSION = ".pt";
    public static final String PRETRAINED_URL_FORMAT = "https://github.com/ultralytics/assets/releases/download/v8.4.0/%s";

    private static final String[][] PRETRAINED_MODELS = new String[][] {
            {"YOLO26n", "yolo26n.pt"},
            {"YOLO26m", "yolo26m.pt"},
            {"YOLO26x", "yolo26x.pt"}
    };

    private static final Map<String, Long> PRETRAINED_WEIGHTS_SIZE;
    static {
        PRETRAINED_WEIGHTS_SIZE = new HashMap<String, Long>();
        PRETRAINED_WEIGHTS_SIZE.put("yolo26n.pt", 5_544_453L);
        PRETRAINED_WEIGHTS_SIZE.put("yolo26m.pt", 44_255_705L);
        PRETRAINED_WEIGHTS_SIZE.put("yolo26x.pt", 118_667_365L);
    }

    private YoloModelRegistry() {}

    public static LinkedHashMap<String, String> buildModelEntries(String modelsDir) {
        LinkedHashMap<String, String> models = new LinkedHashMap<String, String>();
        File yoloDir = modelsDir == null ? new File(YOLO_MODELS_SUBDIR) : new File(modelsDir, YOLO_MODELS_SUBDIR);

        for (String[] pretrained : PRETRAINED_MODELS) {
            models.put("[Pretrained] " + pretrained[0], new File(yoloDir, pretrained[1]).getAbsolutePath());
        }

        File[] customModels = yoloDir.listFiles(file -> file.isFile()
                && file.getName().toLowerCase().endsWith(YOLO_WEIGHTS_EXTENSION)
                && !isPretrainedWeightsFile(file.getName()));
        if (customModels == null) {
            return models;
        }
        Arrays.sort(customModels, Comparator.comparing(File::getName, String.CASE_INSENSITIVE_ORDER));
        for (File modelFile : customModels) {
            models.put("[Custom] " + removeWeightsExtension(modelFile.getName()), modelFile.getAbsolutePath());
        }
        return models;
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
        if (fileName.toLowerCase().endsWith(YOLO_WEIGHTS_EXTENSION)) {
            return fileName.substring(0, fileName.length() - YOLO_WEIGHTS_EXTENSION.length());
        }
        return fileName;
    }
}
