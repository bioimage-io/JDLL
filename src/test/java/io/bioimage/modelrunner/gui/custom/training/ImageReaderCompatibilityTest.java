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
package io.bioimage.modelrunner.gui.custom.training;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.LinkedHashMap;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.JComponent;
import javax.swing.SwingUtilities;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.gui.custom.CellposePluginUI;
import io.bioimage.modelrunner.gui.custom.ConsumerInterface;
import io.bioimage.modelrunner.gui.custom.CrossGoosePluginUI;
import io.bioimage.modelrunner.gui.custom.StarDistPluginUI;
import io.bioimage.modelrunner.gui.custom.UNetPluginUI;
import io.bioimage.modelrunner.gui.custom.YOLOPluginUI;
import io.bioimage.modelrunner.model.detection.Detection;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/** Test the actual disk-entry readers without constructing a GUI or loading a model. */
public class ImageReaderCompatibilityTest {
    @Rule public TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void inferenceReadersDelegateWithoutConvertingPixelsOrDroppingCztAxes() throws Exception {
        ImageConsumer consumer = new ImageConsumer();
        Object[] interfaces = new Object[5];
        SwingUtilities.invokeAndWait(() -> {
            interfaces[0] = new CellposePluginUI(consumer, null);
            interfaces[1] = new StarDistPluginUI(consumer, null);
            interfaces[2] = new UNetPluginUI(consumer, null);
            interfaces[3] = new CrossGoosePluginUI(consumer, null);
            interfaces[4] = new YOLOPluginUI(consumer, null);
        });
        try {
            for (Object ui : interfaces) {
                Class<?> owner = ui instanceof CrossGoosePluginUI ? UNetPluginUI.class : ui.getClass();
                Method read = owner.getDeclaredMethod("readImageFileAsRai", File.class);
                read.setAccessible(true);
                File source = new File(folder.getRoot(), "host-decoded.ome.tif");
                for (RandomAccessibleInterval<?> image : new RandomAccessibleInterval<?>[] {
                        ArrayImgs.unsignedShorts(new short[] {1000, 1001, 2000, 2001, 3000, 3001}, 2, 1, 3, 1, 1),
                        ArrayImgs.doubles(3, 2, 4, 5, 6)}) {
                    consumer.image = image;
                    int previous = consumer.calls;
                    assertSame(image, read.invoke(ui, source));
                    assertSame(source, consumer.file);
                    assertEquals(previous + 1, consumer.calls);
                    assertEquals(image.dimension(2), consumer.getImageFileChannels(source));
                }
                consumer.error = new IOException("Host could not decode this image");
                try {
                    read.invoke(ui, source);
                    fail("Host read failure must not trigger a fallback RGB reader");
                } catch (InvocationTargetException error) {
                    assertSame(consumer.error, error.getCause());
                } finally {
                    consumer.error = null;
                }
            }
        } finally {
            for (Object ui : interfaces) ui.getClass().getMethod("close").invoke(ui);
        }
    }

    @Test
    public void trainingMaskReadersPreserveSixteenBitLabelsWithWrongExtensions() throws Exception {
        for (String format : new String[] {"png", "tiff"}) {
            File mask = new File(folder.getRoot(), format + "-mask." + (format.equals("png") ? "tif" : "png"));
            BufferedImage labels = new BufferedImage(12, 11, BufferedImage.TYPE_USHORT_GRAY);
            labels.getRaster().setSample(1, 1, 0, 1);
            labels.getRaster().setSample(3, 3, 0, 257);
            labels.getRaster().setSample(6, 6, 0, 65535);
            assertTrue(ImageIO.write(labels, format, mask));
            Method stats = SegmentationDatasetPreparer.class.getDeclaredMethod("readMaskStats", File.class);
            stats.setAccessible(true);
            Object result = stats.invoke(null, mask);
            java.lang.reflect.Field count = result.getClass().getDeclaredField("objectCount");
            count.setAccessible(true);
            assertEquals(3, count.getInt(result));
            Method boxes = Class.forName("io.bioimage.modelrunner.gui.custom.yolo.YoloDatasetPreparer")
                    .getDeclaredMethod("readMaskSample", File.class, File.class);
            boxes.setAccessible(true);
            Object sample = boxes.invoke(null, mask, mask);
            java.lang.reflect.Field objects = sample.getClass().getDeclaredField("boxes");
            objects.setAccessible(true);
            assertEquals(3, ((java.util.List<?>) objects.get(sample)).size());
        }
    }

    private class ImageConsumer extends ConsumerInterface {
        RandomAccessibleInterval<?> image;
        File file;
        IOException error;
        int calls;

        @Override @SuppressWarnings("unchecked")
        public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> openImageAsRai(File source)
                throws IOException {
            file = source;
            calls++;
            if (error != null) throw error;
            return (RandomAccessibleInterval<T>) image;
        }

        @Override public String getModelsDir() { return folder.getRoot().getAbsolutePath(); }
        @Override public void updateGUI() {}
        @Override public void setComponents(List<JComponent> components) {}
        @Override public void setVarNames(List<String> names) {}
        @Override public void notifyParams(LinkedHashMap<String, String> args) {}
        @Override public Object getFocusedImage() { return null; }
        @Override public String getFocusedImageName() { return null; }
        @Override public Integer getFocusedImageChannels() { return 1; }
        @Override public Integer getFocusedImageSlices() { return 1; }
        @Override public Integer getFocusedImageFrames() { return 1; }
        @Override public Integer getFocusedImageWidth() { return 1; }
        @Override public Integer getFocusedImageHeight() { return 1; }
        @Override public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getFocusedImageAsRai() { return null; }
        @Override public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> convertIntoRai(Object image) { return null; }
        @Override public <T extends RealType<T> & NativeType<T>> void displayImage(RandomAccessibleInterval<T> image, String axes, String name) {}
        @Override public void displayDetections(List<Detection> detections) {}
    }
}
