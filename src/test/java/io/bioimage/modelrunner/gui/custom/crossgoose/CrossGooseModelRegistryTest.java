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

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.util.LinkedHashMap;
import java.util.Map;

import javax.imageio.ImageIO;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer;
import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer.Framework;
import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer.PreparedDataset;

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
        assertEquals(checkpoint.getAbsoluteFile(), new CrossGooseBackend().findModelFile(checkpoint));
    }

    @Test
    public void trainingUsesTheSelectedFinalDirectoryAndNeverOverwritesAnExistingRun() throws Exception {
        File models = temporaryFolder.newFolder("models");
        CrossGooseTrainingConfig first = CrossGooseTrainingConfig.fromUi("cells", "dataset", 2,
                false, null, "default", models.getAbsolutePath(), "cpu");
        File output = new File(first.getOutputModelDir());
        assertTrue(output.mkdirs());
        Map<String, Object> values = CrossGooseTrainingService.pythonConfig(first, temporaryFolder.getRoot());
        assertEquals(output.getAbsolutePath(), values.get("model_dir"));
        assertEquals(output.getParent(), values.get("output_dir"));
        assertEquals(0, values.get("num_workers"));
        CrossGooseTrainingConfig next = CrossGooseTrainingConfig.fromUi("cells", "dataset", 2,
                true, "source.ckpt", null, models.getAbsolutePath(), "cpu");
        assertEquals("cells-1", next.getModelName());
        assertEquals("source.ckpt", CrossGooseTrainingService.pythonConfig(next,
                temporaryFolder.getRoot()).get("base_model"));
    }

    @Test
    public void normalizesOnlyLayoutsAndNamesTheCrossGooseBackendCannotRead() throws Exception {
        for (String layout : new String[] {"mixed", "root", "canonical", "aliases", "images-first"}) {
            File root = temporaryFolder.newFolder(layout);
            File images = "mixed".equals(layout) ? root : new File(root,
                    "root".equals(layout) ? "images" : "aliases".equals(layout) ? "train/imgs"
                    : "images-first".equals(layout) ? "images/train" : "train/images");
            File masks = "mixed".equals(layout) ? root : new File(root,
                    "root".equals(layout) ? "masks" : "aliases".equals(layout) ? "train/labels"
                    : "images-first".equals(layout) ? "masks/train" : "train/masks");
            images.mkdirs();
            masks.mkdirs();
            BufferedImage pixels = new BufferedImage(16, 16, BufferedImage.TYPE_BYTE_GRAY);
            pixels.getRaster().setSample(4, 4, 0, 1);
            String imageName = "canonical".equals(layout) ? "sample_img.png" : "sample_raw.png";
            File source = new File(images, imageName);
            ImageIO.write(pixels, "png", source);
            ImageIO.write(pixels, "png", new File(masks, "sample_masks.png"));
            PreparedDataset result = SegmentationDatasetPreparer.prepare(root.getAbsolutePath(), "cells",
                    temporaryFolder.newFolder("models-" + layout).getAbsolutePath(), .15,
                    Framework.CROSS_GOOSE, null);
            assertEquals(!"canonical".equals(layout), result.isGenerated());
            File target = new File(result.getDatasetRoot(), "train/images/"
                    + (result.isGenerated() ? "sample.png" : imageName));
            assertTrue(target.isFile());
            assertTrue(Files.isSameFile(source.toPath(), target.toPath()));
        }
    }
}
