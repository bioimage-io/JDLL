/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * #L%
 */
package io.bioimage.modelrunner.gui.custom;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import java.util.regex.Pattern;

import io.bioimage.modelrunner.gui.custom.yolo.YoloImageFiles;

/** One filesystem inference execution: isolated outputs and unambiguous source names. */
public final class InferenceRunFiles {
    private static final Pattern ANNOTATION = Pattern.compile(
            ".*[_-](?:masks?|labels?|gt|seg|segmentation|probabilities|probability|flows?|outlines?|cp)"
            + "(?:[_-][0-9]+)*$", Pattern.CASE_INSENSITIVE);

    private final File directory;
    private final Map<File, String> names = new HashMap<>();

    private InferenceRunFiles(File directory, List<File> images) {
        this.directory = directory;
        Map<String, Integer> counts = new HashMap<>();
        for (File image : images) counts.merge(stem(image.getName()).toLowerCase(Locale.ROOT), 1, Integer::sum);
        Set<String> used = new HashSet<>();
        for (File image : images) {
            String base = stem(image.getName());
            if (counts.get(base.toLowerCase(Locale.ROOT)) > 1) base = image.getName();
            String unique = base;
            for (int n = 1; !used.add(unique.toLowerCase(Locale.ROOT)); n++) unique = base + "-" + n;
            names.put(image.getAbsoluteFile(), unique);
        }
    }

    /** Atomically reserves a fresh run; existing results are never reused or overwritten. */
    public static InferenceRunFiles create(File source, String workflow, String model,
            List<File> images, Consumer<String> log) throws IOException {
        File absolute = source.toPath().toAbsolutePath().normalize().toFile();
        File inputDirectory = absolute.isDirectory() ? absolute : absolute.getParentFile();
        File parent = inputDirectory == null ? null : inputDirectory.getParentFile();
        if (parent == null) throw new IOException("Cannot create a sibling results folder for: " + source);
        Path root = new File(parent, inputDirectory.getName() + "_deepicy_runs").toPath();
        Files.createDirectories(root);
        String base = safeName(workflow) + "_" + safeName(modelName(model));
        Path run;
        for (int n = 0; ; n++) {
            run = root.resolve(base + (n == 0 ? "" : "-" + n));
            try {
                Files.createDirectory(run);
                break;
            } catch (FileAlreadyExistsException occupied) {
                // Another execution (or a file with this name) already owns this destination.
            }
        }
        if (log != null) {
            int excluded = excludedImageCount(source);
            if (excluded > 0) log.accept("Excluded " + excluded + " annotation/generated image(s) from folder inference.");
            log.accept("Results directory: " + run.toAbsolutePath());
        }
        return new InferenceRunFiles(run.toFile(), images);
    }

    public File getDirectory() { return directory; }

    public File outputFile(File image, String suffix) {
        String name = names.get(image.getAbsoluteFile());
        if (name == null) throw new IllegalArgumentException("Image is not part of this inference run: " + image);
        return new File(directory, name + suffix);
    }

    /** Only automatic folder discovery uses this exclusion; explicit files remain selectable. */
    public static boolean isAnnotationOrOutput(File file) {
        return ANNOTATION.matcher(stem(file.getName())).matches();
    }

    private static int excludedImageCount(File source) {
        if (!source.isDirectory()) return 0;
        File[] files = source.listFiles(file -> YoloImageFiles.hasSupportedImageExtension(file)
                && isAnnotationOrOutput(file));
        return files == null ? 0 : files.length;
    }

    private static String modelName(String model) {
        File path = new File(model);
        if (path.isDirectory()) return path.getName();
        String name = stem(path.getName());
        if (name.equalsIgnoreCase("model") || name.matches("(?i)(?:weights[_-])?(?:best|last)")) {
            File parent = path.getAbsoluteFile().getParentFile();
            if (parent != null && (parent.getName().equalsIgnoreCase("weights")
                    || parent.getName().equalsIgnoreCase("checkpoints"))) parent = parent.getParentFile();
            if (parent != null) return parent.getName();
        }
        return name;
    }

    private static String safeName(String name) {
        String safe = name.replaceAll("[^\\p{L}\\p{N}._-]+", "-")
                .replaceAll("^[.-]+|[.-]+$", "");
        if (safe.isEmpty()) return "model";
        return safe.substring(0, Math.min(safe.length(), 100));
    }

    private static String stem(String name) {
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }
}
