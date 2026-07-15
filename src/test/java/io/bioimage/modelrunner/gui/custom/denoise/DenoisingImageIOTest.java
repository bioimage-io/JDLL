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
package io.bioimage.modelrunner.gui.custom.denoise;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.awt.image.BufferedImage;
import java.io.File;

import javax.imageio.ImageIO;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import net.imglib2.RandomAccess;
import net.imglib2.type.numeric.real.FloatType;

public class DenoisingImageIOTest {
    @Rule public TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void readsSixteenBitImageAndUsesCollisionSafeNames() throws Exception {
        File input = new File(folder.getRoot(), "sample.tif");
        BufferedImage image = new BufferedImage(5, 4, BufferedImage.TYPE_USHORT_GRAY);
        image.getRaster().setSample(2, 1, 0, 1234);
        assertTrue(ImageIO.write(image, "tiff", input));
        DenoisingImageIO.ImageData data = DenoisingImageIO.read(input);
        assertEquals("xy", data.getAxes());
        assertEquals(16, data.getBits());
        RandomAccess<FloatType> access = data.getImage().randomAccess();
        access.setPosition(new long[] {2, 1});
        assertEquals(1234.0d, access.get().getRealDouble(), 0.0d);

        File first = DenoisingImageIO.outputFile(input);
        assertEquals("sample_denoised.tif", first.getName());
        assertTrue(first.createNewFile());
        assertEquals("sample_denoised-1.tif", DenoisingImageIO.outputFile(input).getName());
    }

    @Test
    public void publishesCompletedFolderAndRemovesTemporaryTrees() throws Exception {
        File source = folder.newFolder("images");
        File output = DenoisingImageIO.outputDirectory(source);
        File temporary = DenoisingImageIO.temporaryDirectoryFor(output);
        assertTrue(temporary.mkdirs());
        assertTrue(new File(temporary, "result.txt").createNewFile());
        DenoisingImageIO.publishDirectory(temporary, output);
        assertTrue(new File(output, "result.txt").isFile());

        File abandoned = DenoisingImageIO.temporaryDirectoryFor(output);
        assertTrue(new File(abandoned, "nested").mkdirs());
        assertTrue(new File(abandoned, "nested/result.txt").createNewFile());
        DenoisingImageIO.deleteRecursively(abandoned);
        assertTrue(!abandoned.exists());
    }
}
