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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.util.LinkedHashMap;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class CrossGooseModelRegistryTest {

    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void discoversOnlyCompleteModelDirectories() throws Exception {
        File temporaryDirectory = temporaryFolder.getRoot();
        File root = new File(temporaryDirectory, CrossGooseModelRegistry.MODELS_SUBDIR);
        File bundled = new File(root, "bundled");
        File trained = new File(root, "trained");
        File incomplete = new File(root, "incomplete");
        Files.createDirectories(bundled.toPath());
        Files.createDirectories(new File(trained, "checkpoints").toPath());
        Files.createDirectories(incomplete.toPath());
        Files.write(new File(bundled, "config.yaml").toPath(), new byte[] {1});
        Files.write(new File(bundled, "weights.ckpt").toPath(), new byte[] {1});
        Files.write(new File(trained, "config.yaml").toPath(), new byte[] {1});
        Files.write(new File(trained, "checkpoints/last.ckpt").toPath(), new byte[] {1});
        Files.write(new File(incomplete, "weights.ckpt").toPath(), new byte[] {1});

        LinkedHashMap<String, String> entries = CrossGooseModelRegistry.buildModelEntries(
                temporaryDirectory.getAbsolutePath());

        assertEquals(2, entries.size());
        assertTrue(entries.containsValue(bundled.getAbsolutePath()));
        assertTrue(entries.containsValue(trained.getAbsolutePath()));
        assertFalse(entries.containsValue(incomplete.getAbsolutePath()));
    }

    @Test
    public void acceptsCheckpointWhenItsModelDirectoryHasConfig() throws Exception {
        File temporaryDirectory = temporaryFolder.getRoot();
        File model = new File(temporaryDirectory, "model");
        File checkpoints = new File(model, "checkpoints");
        Files.createDirectories(checkpoints.toPath());
        File checkpoint = new File(checkpoints, "last.ckpt");
        Files.write(new File(model, "config.yaml").toPath(), new byte[] {1});
        Files.write(checkpoint.toPath(), new byte[] {1});

        assertTrue(CrossGooseModelRegistry.isModelPath(model.getAbsolutePath()));
        assertTrue(CrossGooseModelRegistry.isModelPath(checkpoint.getAbsolutePath()));
        assertEquals(model.getAbsoluteFile(), CrossGooseModelRegistry.modelDirectory(checkpoint));
    }
}
