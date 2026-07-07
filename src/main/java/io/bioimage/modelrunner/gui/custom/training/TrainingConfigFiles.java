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
package io.bioimage.modelrunner.gui.custom.training;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import io.bioimage.modelrunner.utils.JSONUtils;

public final class TrainingConfigFiles {

    public static final String CONFIG_FILE_NAME = "config.json";

    private TrainingConfigFiles() {}

    /**
     * Returns the config file matching a model name.
     *
     * @param modelsDir the models root.
     * @param familySubdir the model family subdirectory.
     * @param modelName the model name.
     * @return the config file.
     */
    public static File configFileForModelName(String modelsDir, String familySubdir, String modelName) {
        if (modelName == null || modelName.trim().isEmpty()) {
            return null;
        }
        File familyDir = modelsDir == null ? new File(familySubdir) : new File(modelsDir, familySubdir);
        return new File(new File(familyDir, modelName.trim()), CONFIG_FILE_NAME);
    }

    /**
     * Returns whether value points to a config JSON file.
     *
     * @param value the value.
     * @return true if value points to a config JSON file.
     */
    public static boolean isConfigPath(String value) {
        if (value == null || value.trim().isEmpty()) {
            return false;
        }
        File file = new File(value.trim());
        return file.isFile() && CONFIG_FILE_NAME.equalsIgnoreCase(file.getName());
    }

    /**
     * Loads a config map.
     *
     * @param configFile the config file.
     * @return the loaded map, or null.
     */
    public static Map<String, Object> load(File configFile) {
        if (configFile == null || !configFile.isFile()) {
            return null;
        }
        try {
            return JSONUtils.load(configFile.getAbsolutePath());
        } catch (IOException e) {
            return null;
        }
    }

    /**
     * Loads a config map.
     *
     * @param configPath the config path.
     * @return the loaded map, or null.
     */
    public static Map<String, Object> load(String configPath) {
        return configPath == null ? null : load(new File(configPath.trim()));
    }

    /**
     * Returns a nested map value.
     *
     * @param root the root map.
     * @param keys the nested keys.
     * @return the nested map, or null.
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> mapAt(Map<String, Object> root, String... keys) {
        Object value = objectAt(root, keys);
        return value instanceof Map ? (Map<String, Object>) value : null;
    }

    /**
     * Returns a nested object value.
     *
     * @param root the root map.
     * @param keys the nested keys.
     * @return the nested object, or null.
     */
    public static Object objectAt(Map<String, Object> root, String... keys) {
        Object current = root;
        for (String key : keys) {
            if (!(current instanceof Map)) {
                return null;
            }
            current = ((Map<?, ?>) current).get(key);
        }
        return current;
    }

    /**
     * Returns a nested string value.
     *
     * @param root the root map.
     * @param keys the nested keys.
     * @return the string, or null.
     */
    public static String stringAt(Map<String, Object> root, String... keys) {
        Object value = objectAt(root, keys);
        return value == null ? null : value.toString().trim();
    }
}
