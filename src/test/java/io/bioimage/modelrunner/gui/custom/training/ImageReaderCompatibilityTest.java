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
import static org.junit.Assert.assertTrue;

import java.awt.image.BufferedImage;
import java.io.File;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.util.Iterator;

import javax.imageio.ImageIO;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.gui.custom.CellposePluginUI;
import io.bioimage.modelrunner.gui.custom.StarDistPluginUI;
import io.bioimage.modelrunner.gui.custom.UNetPluginUI;
import io.bioimage.modelrunner.gui.custom.YOLOPluginUI;
import io.bioimage.modelrunner.gui.custom.denoise.DenoisingImageIO;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

/** Test the actual disk-entry readers without constructing a GUI or loading a model. */
public class ImageReaderCompatibilityTest {
    @Rule public TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void inferenceReadersDecodeByContentsRatherThanExtension() throws Exception {
        for (String format : new String[] {"png", "tiff", "jpeg", "bmp"}) {
            File original = new File(folder.getRoot(), "original." + format);
            BufferedImage image = new BufferedImage(12, 11, BufferedImage.TYPE_3BYTE_BGR);
            image.setRGB(5, 4, 0x123456);
            assertTrue(ImageIO.write(image, format, original));
            for (String extension : new String[] {"png", "tif", "bmp", "jpg"}) {
                File renamed = new File(folder.getRoot(), format + "-renamed." + extension);
                Files.copy(original.toPath(), renamed.toPath());
                for (Class<?> ui : new Class<?>[] {CellposePluginUI.class, StarDistPluginUI.class,
                        UNetPluginUI.class, YOLOPluginUI.class}) {
                    Method read = ui.getDeclaredMethod("readImageFileAsRai", File.class);
                    read.setAccessible(true);
                    assertSamePixels((RandomAccessibleInterval<?>) read.invoke(null, original),
                            (RandomAccessibleInterval<?>) read.invoke(null, renamed));
                }
                assertSamePixels(DenoisingImageIO.read(original).getImage(),
                        DenoisingImageIO.read(renamed).getImage());
            }
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

    private static void assertSamePixels(RandomAccessibleInterval<?> expected, RandomAccessibleInterval<?> actual) {
        assertEquals(expected.numDimensions(), actual.numDimensions());
        for (int d = 0; d < expected.numDimensions(); d++) assertEquals(expected.dimension(d), actual.dimension(d));
        Iterator<?> values = Views.flatIterable(expected).iterator();
        for (Object value : Views.flatIterable(actual)) {
            assertEquals(((RealType<?>) values.next()).getRealDouble(), ((RealType<?>) value).getRealDouble(), 0);
        }
    }
}
