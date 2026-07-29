/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * #L%
 */
package io.bioimage.modelrunner.gui.custom.crossgoose;

import java.io.File;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Locale;

/** Discovers Cross-GOOSE model directories and checkpoints. */
public final class CrossGooseModelRegistry {

    public static final String MODELS_SUBDIR = "cross-goose";
    public static final String DEFAULT_ARCHITECTURE = "default";

    private CrossGooseModelRegistry() {}

    public static LinkedHashMap<String, String> buildModelEntries(String modelsDir) {
        LinkedHashMap<String, String> models = new LinkedHashMap<String, String>();
        File root = modelsDir == null ? new File(MODELS_SUBDIR) : new File(modelsDir, MODELS_SUBDIR);
        File[] directories = root.listFiles(CrossGooseModelRegistry::isModelDirectory);
        if (directories == null) return models;
        Arrays.sort(directories, Comparator.comparing(File::getName, String.CASE_INSENSITIVE_ORDER));
        for (File directory : directories) {
            models.put("[Custom] " + directory.getName(), directory.getAbsolutePath());
        }
        return models;
    }

    public static LinkedHashMap<String, String> scratchArchitectures() {
        LinkedHashMap<String, String> architectures = new LinkedHashMap<String, String>();
        architectures.put("Default", DEFAULT_ARCHITECTURE);
        return architectures;
    }

    public static boolean isKnownScratchArchitecture(String value) {
        return value != null && DEFAULT_ARCHITECTURE.equalsIgnoreCase(value.trim());
    }

    public static boolean isModelPath(String path) {
        if (path == null || path.trim().isEmpty()) return false;
        File file = new File(path.trim());
        return file.isDirectory() ? isModelDirectory(file)
                : file.isFile() && isCheckpoint(file) && modelDirectory(file) != null;
    }

    public static File modelDirectory(File path) {
        if (path == null) return null;
        File directory = path.isDirectory() ? path : path.getParentFile();
        if (directory != null && "checkpoints".equals(directory.getName())) {
            directory = directory.getParentFile();
        }
        return isModelDirectory(directory) ? directory.getAbsoluteFile() : null;
    }

    public static String removeCheckpointExtension(String name) {
        if (name == null) return "";
        String lower = name.toLowerCase(Locale.ROOT);
        return lower.endsWith(".ckpt") ? name.substring(0, name.length() - 5) : name;
    }

    private static boolean isModelDirectory(File directory) {
        if (directory == null || !directory.isDirectory() || !hasConfig(directory)) return false;
        File bundledWeights = new File(directory, "weights.ckpt");
        if (bundledWeights.isFile()) return true;
        File checkpoints = new File(directory, "checkpoints");
        File[] files = checkpoints.listFiles(CrossGooseModelRegistry::isCheckpoint);
        return files != null && files.length > 0;
    }

    private static boolean hasConfig(File directory) {
        return new File(directory, "config.yaml").isFile()
                || new File(directory, "config.yml").isFile()
                || new File(directory, "config.json").isFile();
    }

    private static boolean isCheckpoint(File file) {
        return file != null && file.isFile()
                && file.getName().toLowerCase(Locale.ROOT).endsWith(".ckpt");
    }
}
