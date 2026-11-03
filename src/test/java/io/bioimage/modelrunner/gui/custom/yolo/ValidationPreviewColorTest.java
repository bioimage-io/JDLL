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
package io.bioimage.modelrunner.gui.custom.yolo;

import static org.junit.Assert.*;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import javax.imageio.ImageIO;
import javax.swing.JButton;
import javax.swing.SwingUtilities;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import io.bioimage.modelrunner.gui.custom.unet.DenseValidationPreviewPanel;

public class ValidationPreviewColorTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test public void pngMasksCanBeRecoloredWithoutReloadingOrMovingTheViewport() throws Exception {
        BufferedImage source = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);
        source.setRGB(0, 0, Color.WHITE.getRGB());
        BufferedImage prediction = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);
        prediction.setRGB(1, 1, new Color(37, 73, 109).getRGB());
        File image = temporary.newFile("image.png"), mask = temporary.newFile("prediction.png");
        ImageIO.write(source, "png", image);
        ImageIO.write(prediction, "png", mask);
        JsonObject item = new JsonObject();
        item.addProperty("image_path", image.toString());
        item.addProperty("prediction_path", mask.toString());
        // A precolored overlay must not take precedence over the editable assets.
        item.addProperty("overlay_path", temporary.newFile("unused-overlay.png").toString());
        JsonArray items = new JsonArray();
        items.add(item);
        items.add(item.deepCopy());
        JsonObject manifest = new JsonObject();
        manifest.addProperty("epoch", 1);
        manifest.addProperty("task", "binary_semantic");
        manifest.add("items", items);
        File json = temporary.newFile("preview.json");
        Files.write(json.toPath(), manifest.toString().getBytes(StandardCharsets.UTF_8));

        SwingUtilities.invokeAndWait(() -> {
            try {
                DenseValidationPreviewPanel panel = new DenseValidationPreviewPanel();
                panel.loadPreview(json.toString());
                YoloValidationPreviewPanel raster = (YoloValidationPreviewPanel) field(panel, "raster");
                JButton picker = (JButton) field(raster, "annotationColorButton");
                assertTrue("Masks need the same color picker as boxes", picker.isVisible() && picker.isEnabled());
                YoloImageDisplayPanel display = (YoloImageDisplayPanel) field(raster, "imagePanel");
                BufferedImage loadedSource = (BufferedImage) field(raster, "currentImage");
                BufferedImage loadedMask = (BufferedImage) field(raster, "currentPrediction");
                assertEquals(new Color(37, 73, 109), picker.getForeground());
                assertEquals(0x0e1d2b, overlay(loadedSource, loadedMask, null).getRGB(1, 1) & 0xffffff);
                display.setViewport(new YoloImageDisplayPanel.Viewport(2.5, 12, 24, true));
                selectColor(raster, Color.CYAN);
                assertEquals(Color.CYAN, picker.getForeground());
                assertSame("Recoloring must not reread the source", loadedSource, field(raster, "currentImage"));
                assertSame("Recoloring must not reread predictions", loadedMask, field(raster, "currentPrediction"));
                assertEquals(2.5, display.getViewport().getZoom(), 0);
                assertEquals(12, display.getViewport().getPanX(), 0);
                assertEquals(24, display.getViewport().getPanY(), 0);
                assertTrue(display.getViewport().isExpandedToFill());
                BufferedImage tinted = overlay(loadedSource, loadedMask, Color.CYAN);
                assertEquals(0x006666, tinted.getRGB(1, 1) & 0xffffff);
                BufferedImage displayed = (BufferedImage) field(field(display, "previewSource"), "image");
                assertEquals(tinted.getRGB(1, 1), displayed.getRGB(1, 1));
                assertEquals(Color.WHITE.getRGB(), tinted.getRGB(0, 0));
                assertEquals("Original assets stay untouched", 0xff25496d, loadedMask.getRGB(1, 1));
                ((JButton) field(raster, "nextButton")).doClick();
                assertEquals(Color.CYAN, field(raster, "selectedMaskColor"));
                panel.loadPreview(json.toString());
                assertEquals(Color.CYAN, field(raster, "selectedMaskColor"));
                panel.clearPreview();
                assertFalse(picker.isVisible());
                assertNull(field(raster, "currentImage"));
                assertNull(field(raster, "currentPrediction"));
                panel.loadPreview(json.toString());
                assertEquals("Keep the color for the next run", Color.CYAN, field(raster, "selectedMaskColor"));
                panel.clearPreview();
            } catch (Exception e) {
                throw new AssertionError(e);
            }
        });
    }

    @Test public void overlayOnlyLegacyPreviewsStillLoadAndEmptyMasksRemainEditable() throws Exception {
        BufferedImage source = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
        source.setRGB(0, 0, Color.WHITE.getRGB());
        File image = temporary.newFile("legacy.png");
        ImageIO.write(source, "png", image);
        JsonObject item = new JsonObject();
        item.addProperty("overlay_path", image.toString());
        JsonArray items = new JsonArray();
        items.add(item);
        File emptyMask = temporary.newFile("empty.png");
        ImageIO.write(new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB), "png", emptyMask);
        JsonObject empty = new JsonObject();
        empty.addProperty("image_path", image.toString());
        empty.addProperty("prediction_path", emptyMask.toString());
        items.add(empty);
        JsonObject manifest = new JsonObject();
        manifest.add("items", items);
        File json = temporary.newFile("legacy.json");
        Files.write(json.toPath(), manifest.toString().getBytes(StandardCharsets.UTF_8));
        SwingUtilities.invokeAndWait(() -> {
            try {
                YoloValidationPreviewPanel panel = new YoloValidationPreviewPanel();
                panel.loadPreview(json.toString());
                assertNotNull(field(panel, "currentImage"));
                assertNull(field(panel, "currentPrediction"));
                assertFalse(((JButton) field(panel, "annotationColorButton")).isVisible());
                ((JButton) field(panel, "nextButton")).doClick();
                assertTrue(((JButton) field(panel, "annotationColorButton")).isVisible());
                selectColor(panel, Color.RED);
                YoloImageDisplayPanel display = (YoloImageDisplayPanel) field(panel, "imagePanel");
                BufferedImage displayed = (BufferedImage) field(field(display, "previewSource"), "image");
                assertEquals(Color.WHITE.getRGB(), displayed.getRGB(0, 0));
                panel.clearPreview();
            } catch (Exception e) {
                throw new AssertionError(e);
            }
        });
    }

    private static BufferedImage overlay(BufferedImage source, BufferedImage mask, Color color) throws Exception {
        Method method = YoloValidationPreviewPanel.class.getDeclaredMethod("maskOverlay",
                BufferedImage.class, BufferedImage.class, Color.class);
        method.setAccessible(true);
        return (BufferedImage) method.invoke(null, source, mask, color);
    }

    private static void selectColor(YoloValidationPreviewPanel panel, Color color) throws Exception {
        Method method = YoloValidationPreviewPanel.class.getDeclaredMethod("setSelectedBoxColor", Color.class);
        method.setAccessible(true);
        method.invoke(panel, color);
    }

    private static Object field(Object target, String name) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        return field.get(target);
    }
}
