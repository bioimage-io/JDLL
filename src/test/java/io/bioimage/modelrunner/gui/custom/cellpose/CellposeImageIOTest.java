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
package io.bioimage.modelrunner.gui.custom.cellpose;

import static org.junit.Assert.*;

import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.File;
import java.lang.reflect.Method;
import java.util.List;

import javax.imageio.ImageIO;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.gui.custom.CellposePluginUI;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

public class CellposeImageIOTest {
    @Rule public TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void sixteenBitChannelsKeepOriginalIntensity() throws Exception {
        WritableRaster raster = Raster.createInterleavedRaster(DataBuffer.TYPE_USHORT, 2, 1, 3, null);
        raster.setPixel(0, 0, new int[] {1000, 2000, 3000});
        raster.setPixel(1, 0, new int[] {1001, 2001, 3001});
        BufferedImage image = new BufferedImage(new ComponentColorModel(ColorSpace.getInstance(ColorSpace.CS_sRGB),
                false, false, Transparency.OPAQUE, DataBuffer.TYPE_USHORT), raster, false, null);
        File file = folder.newFile("raw.tif");
        assertTrue(ImageIO.write(image, "TIFF", file));
        RandomAccessibleInterval<FloatType> result = read(file);
        assertArrayEquals(new long[] {2, 1, 3, 1, 1}, result.dimensionsAsLongArray());
        float[] expected = {1000, 1001, 2000, 2001, 3000, 3001};
        int i = 0;
        for (FloatType value : Views.flatIterable(result)) assertEquals(expected[i++], value.get(), 0);
    }

    @Test
    public void savesAllZTPlanesAsFloatLabelsAndDoesNotReprocessOrOverwriteThem() throws Exception {
        File source = folder.newFile("source.png");
        assertTrue(ImageIO.write(new BufferedImage(2, 2, BufferedImage.TYPE_BYTE_GRAY), "png", source));
        float[] values = {1, 70000, 90000, 120000, 130000, 140000};
        Tensor<FloatType> labels = Tensor.build("labels", "xyzt", ArrayImgs.floats(values, 1, 1, 2, 3));
        File output = (File) method("maskOutputFileFor", File.class).invoke(null, source);
        method("writeLabelMask", Tensor.class, File.class).invoke(null, labels, output);
        RandomAccessibleInterval<FloatType> read = read(output);
        assertArrayEquals(new long[] {1, 1, 1, 6, 1}, read.dimensionsAsLongArray());
        int i = 0;
        for (FloatType value : Views.flatIterable(read)) assertEquals(values[i++], value.get(), 0);
        File next = (File) method("maskOutputFileFor", File.class).invoke(null, source);
        assertNotEquals(output, next);
        method("writeLabelMask", Tensor.class, File.class).invoke(null, labels, next);
        List<?> inputs = (List<?>) method("systemImages", File.class).invoke(null, folder.getRoot());
        assertEquals(1, inputs.size());
        assertEquals(source, inputs.get(0));
    }

    @SuppressWarnings("unchecked")
    private static RandomAccessibleInterval<FloatType> read(File file) throws Exception {
        return (RandomAccessibleInterval<FloatType>) method("readImageFileAsRai", File.class).invoke(null, file);
    }

    private static Method method(String name, Class<?>... parameters) throws Exception {
        Method method = CellposePluginUI.class.getDeclaredMethod(name, parameters);
        method.setAccessible(true);
        return method;
    }
}
