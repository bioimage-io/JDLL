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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.FloatType;

public class DenoisingImagesTest {
    @Test
    public void blendsInOriginalTypeAndIntensityDomain() {
        RandomAccessibleInterval<UnsignedShortType> original = ArrayImgs.unsignedShorts(new short[] {10, 20}, 2, 1);
        RandomAccessibleInterval<FloatType> denoised = ArrayImgs.floats(new float[] {30, 60}, 2, 1);
        RandomAccessibleInterval<UnsignedShortType> blend = DenoisingImages.blend(original, denoised, 0.25d);
        assertEquals(15, blend.randomAccess().get().getInteger());
        RandomAccess<UnsignedShortType> access = blend.randomAccess();
        access.setPosition(1, 0);
        assertEquals(30, access.get().getInteger());
    }

    @Test
    public void extractsRequestedCztPlane() {
        RandomAccessibleInterval<FloatType> image = ArrayImgs.floats(3, 2, 2, 4, 2);
        RandomAccess<FloatType> access = image.randomAccess();
        access.setPosition(new long[] {1, 1, 1, 2, 1});
        access.get().set(77.0f);
        RandomAccessibleInterval<FloatType> plane = DenoisingImages.previewPlane(
                image, "xyczt", 1, 2, 1);
        assertArrayEquals(new long[] {3, 2}, plane.dimensionsAsLongArray());
        RandomAccess<FloatType> planeAccess = plane.randomAccess();
        planeAccess.setPosition(new long[] {1, 1});
        assertEquals(77.0d, planeAccess.get().getRealDouble(), 0.0d);
    }
}
