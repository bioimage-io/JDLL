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

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

/** ImgLib2 operations used by preview and final strength blending. */
public final class DenoisingImages {
    private DenoisingImages() {}

    public static <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> blend(
            RandomAccessibleInterval<T> original, RandomAccessibleInterval<FloatType> denoised,
            double strength) {
        if (!sameDimensions(original, denoised)) {
            throw new IllegalArgumentException("Original and denoised images must have the same dimensions.");
        }
        final double alpha = Math.max(0.0d, Math.min(1.0d, strength));
        T type = Util.getTypeFromInterval(original).createVariable();
        Img<T> result = new ArrayImgFactory<T>(type).create(original.dimensionsAsLongArray());
        LoopBuilder.setImages(Views.zeroMin(original), Views.zeroMin(denoised), result)
                .multiThreaded()
                .forEachPixel((source, filtered, target) -> target.setReal(
                        (1.0d - alpha) * source.getRealDouble() + alpha * filtered.getRealDouble()));
        return result;
    }

    public static <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> previewPlane(
            RandomAccessibleInterval<T> image, String axes, long channel, long z, long time) {
        RandomAccessibleInterval<T> view = image;
        String currentAxes = axes == null ? axesFor(image.numDimensions()) : axes.toLowerCase();
        view = slice(view, currentAxes, 't', time);
        currentAxes = removeAxis(currentAxes, 't');
        view = slice(view, currentAxes, 'z', z);
        currentAxes = removeAxis(currentAxes, 'z');
        view = slice(view, currentAxes, 'c', channel);
        return view;
    }

    public static String axesFor(int dimensions) {
        switch (dimensions) {
            case 2: return "xy";
            case 3: return "xyc";
            case 4: return "xycz";
            case 5: return "xyczt";
            default: throw new IllegalArgumentException("Unsupported image dimensionality: " + dimensions);
        }
    }

    private static <T> RandomAccessibleInterval<T> slice(RandomAccessibleInterval<T> image,
            String axes, char axis, long requested) {
        int index = axes.indexOf(axis);
        if (index < 0) return image;
        long position = Math.max(image.min(index), Math.min(image.max(index), requested));
        return Views.hyperSlice(image, index, position);
    }

    private static String removeAxis(String axes, char axis) {
        int index = axes.indexOf(axis);
        return index < 0 ? axes : axes.substring(0, index) + axes.substring(index + 1);
    }

    private static boolean sameDimensions(RandomAccessibleInterval<?> left, RandomAccessibleInterval<?> right) {
        if (left.numDimensions() != right.numDimensions()) return false;
        for (int d = 0; d < left.numDimensions(); d++) {
            if (left.dimension(d) != right.dimension(d)) return false;
        }
        return true;
    }
}
