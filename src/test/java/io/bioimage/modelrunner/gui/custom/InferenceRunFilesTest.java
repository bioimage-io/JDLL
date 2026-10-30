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

import static org.junit.Assert.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javax.imageio.ImageIO;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.gui.custom.yolo.YoloDetectionCsvWriter;
import io.bioimage.modelrunner.gui.custom.yolo.YoloDetectionGeoJsonWriter;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageFiles;
import io.bioimage.modelrunner.model.detection.Detection;

public class InferenceRunFilesTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test
    public void singleFileAndFolderUseOneSiblingRootAndFreshModelRuns() throws Exception {
        File input = temporary.newFolder("experiment");
        File image = image(input, "sample.png");
        File model = temporary.newFolder("nathaniel-5");
        File weights = new File(model, "model.pt");
        Files.createFile(weights.toPath());
        List<String> log = new ArrayList<>();
        InferenceRunFiles first = InferenceRunFiles.create(input, "unet", weights.toString(),
                Collections.singletonList(image), log::add);
        File root = new File(temporary.getRoot(), "experiment_deepicy_runs");
        assertEquals(new File(root, "unet_nathaniel-5"), first.getDirectory());
        File output = first.outputFile(image, "_labels.tif");
        assertEquals("sample_labels.tif", output.getName());
        Files.write(output.toPath(), new byte[] {7});
        InferenceRunFiles second = InferenceRunFiles.create(new File(input, "./sample.png"), "unet", model.toString(),
                Collections.singletonList(image), null);
        assertEquals(new File(root, "unet_nathaniel-5-1"), second.getDirectory());
        assertArrayEquals(new byte[] {7}, Files.readAllBytes(output.toPath()));
        assertEquals(Collections.singletonList(image), Arrays.asList(input.listFiles()));
        assertTrue(log.get(0).contains(first.getDirectory().toString()));
        assertThrows(IllegalArgumentException.class, () -> first.outputFile(new File(input, "other.png"), "_labels.tif"));
    }

    @Test
    public void concurrentRunsCannotReserveTheSameDirectory() throws Exception {
        File input = temporary.newFolder("images");
        ExecutorService executor = Executors.newFixedThreadPool(4);
        try {
            List<Future<File>> runs = new ArrayList<>();
            for (int i = 0; i < 8; i++) runs.add(executor.submit(() -> InferenceRunFiles.create(
                    input, "cellpose", "cyto3", Collections.emptyList(), null).getDirectory()));
            HashSet<File> directories = new HashSet<>();
            for (Future<File> run : runs) assertTrue(directories.add(run.get()));
            assertEquals(8, directories.size());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    public void outputNamesPreserveExtensionsWhenBasenamesCollide() throws Exception {
        File input = temporary.newFolder("images");
        List<File> images = Arrays.asList(new File(input, "sample.tif"), new File(input, "sample.png"),
                new File(input, "sample.png.tif"), new File(input, "Sample.PNG"));
        InferenceRunFiles run = InferenceRunFiles.create(input, "stardist", "custom", images, null);
        HashSet<String> names = new HashSet<>();
        for (File image : images) assertTrue(names.add(run.outputFile(image, "_labels.tif").getName().toLowerCase()));
        assertEquals("sample.tif_labels.tif", run.outputFile(images.get(0), "_labels.tif").getName());
        assertEquals("sample.png_labels.tif", run.outputFile(images.get(1), "_labels.tif").getName());
    }

    @Test
    public void everySegmentationFolderSelectorSkipsAnnotationsButAllowsExplicitFiles() throws Exception {
        File input = temporary.newFolder("images");
        File original = image(input, "cell.png");
        File annotation = image(input, "cell_MASKS.png");
        for (String name : Arrays.asList("cell_labels-2.png", "cell_label.png", "cell-mask_0001.png",
                "cell_gt.png", "cell_seg.npy", "cell_segmentation.png", "cell_unet_labels.tif",
                "cell_probabilities.png", "cell_flows.png", "cell_outlines.png", "cell_cp.png")) {
            assertTrue(name, InferenceRunFiles.isAnnotationOrOutput(new File(name)));
            Files.write(new File(input, name).toPath(), new byte[] {0});
        }
        File child = new File(input, "other_deepicy_runs");
        assertTrue(child.mkdir());
        image(child, "not-an-input.png");
        assertEquals(Collections.singletonList(original), YoloImageFiles.readableImagesInDirectory(input));
        assertEquals(original, YoloImageFiles.previewImageInDirectory(input));
        assertTrue(YoloImageFiles.isValidDroppedPath(annotation));
        for (Class<?> ui : Arrays.asList(UNetPluginUI.class, StarDistPluginUI.class,
                YOLOPluginUI.class, CellposePluginUI.class)) {
            Method selector = ui.getDeclaredMethod(ui == CellposePluginUI.class ? "systemImages"
                    : "systemInferenceImages", File.class);
            selector.setAccessible(true);
            assertEquals(Collections.singletonList(original), selector.invoke(null, input));
            assertEquals(Collections.singletonList(annotation), selector.invoke(null, annotation));
        }
        List<String> log = new ArrayList<>();
        InferenceRunFiles.create(input, "unet", "test", Collections.singletonList(original), log::add);
        assertTrue(log.get(0).startsWith("Excluded 12 annotation/generated image(s)"));
        assertFalse(InferenceRunFiles.isAnnotationOrOutput(new File("masking_experiment.png")));
        assertFalse(InferenceRunFiles.isAnnotationOrOutput(new File("cell_labelled_image.png")));
        File denoised = image(input, "cell_denoised.png");
        assertTrue(YoloImageFiles.readableImagesInDirectory(input).contains(denoised));
    }

    @Test
    public void yoloExportsStayInRunDirectoryAndKeepOriginalSourceIdentity() throws Exception {
        File input = temporary.newFolder("images");
        File image = image(input, "cell.png");
        File checkpoints = temporary.newFolder("detector", "weights");
        File model = new File(checkpoints, "best.pt");
        Files.createFile(model.toPath());
        InferenceRunFiles run = InferenceRunFiles.create(input, "yolo", model.toString(),
                Collections.singletonList(image), null);
        assertEquals("yolo_detector", run.getDirectory().getName());
        Map<String, List<Detection>> detections = Collections.singletonMap(image.getAbsolutePath(), Collections.emptyList());
        File csv = new File(run.getDirectory(), "detections.csv");
        File geojson = run.outputFile(image, "_detections.geojson");
        YoloDetectionCsvWriter.write(csv, detections);
        YoloDetectionGeoJsonWriter.write(geojson, image.getAbsolutePath(), Collections.emptyList());
        assertTrue(new String(Files.readAllBytes(csv.toPath()), StandardCharsets.UTF_8).contains(image.getAbsolutePath()));
        assertTrue(new String(Files.readAllBytes(geojson.toPath()), StandardCharsets.UTF_8).contains("image_path"));
        assertEquals(Collections.singletonList(image), Arrays.asList(input.listFiles()));
    }

    @Test
    public void occupiedRootFailsWithoutChangingInputsAndModelNamesCannotEscapeRunRoot() throws Exception {
        File input = temporary.newFolder("images");
        File root = temporary.newFile("images_deepicy_runs");
        assertThrows(java.io.IOException.class, () -> InferenceRunFiles.create(
                input, "unet", "model", Collections.emptyList(), null));
        assertTrue(root.delete());
        InferenceRunFiles run = InferenceRunFiles.create(input, "cross-goose", "../../unsafe:model?",
                Collections.emptyList(), null);
        assertEquals(root, run.getDirectory().getParentFile());
        assertFalse(run.getDirectory().getName().contains(":"));
        assertFalse(run.getDirectory().getName().contains("?"));
    }

    private File image(File folder, String name) throws Exception {
        File image = new File(folder, name);
        assertTrue(ImageIO.write(new BufferedImage(2, 2, BufferedImage.TYPE_BYTE_GRAY), "png", image));
        return image;
    }
}
