/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.gui.custom;

import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ComponentColorModel;
import java.awt.image.ComponentSampleModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferFloat;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.util.Iterator;
import java.util.Locale;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageTypeSpecifier;
import javax.imageio.ImageWriter;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.metadata.IIOMetadataNode;
import javax.imageio.stream.ImageOutputStream;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

/** Writes single-channel labels, preserving Z/time and float label precision in TIFF. */
public final class LabelMaskWriter {
    private LabelMaskWriter() {}

    public static <T extends RealType<T> & NativeType<T>> File write(Tensor<T> tensor, File requested)
            throws IOException {
        RandomAccessibleInterval<T> labels = Views.zeroMin(tensor.getData());
        String axes = tensor.getAxesOrderString().toLowerCase(Locale.ROOT);
        int zAxis = axes.indexOf('z'), tAxis = axes.indexOf('t');
        if (tAxis < 0) tAxis = axes.indexOf('b');
        int depth = zAxis < 0 ? 1 : Math.toIntExact(labels.dimension(zAxis));
        int frames = tAxis < 0 ? 1 : Math.toIntExact(labels.dimension(tAxis));
        int count = Math.multiplyExact(depth, frames);
        String name = requested.getName().toLowerCase(Locale.ROOT);
        boolean tiff = count > 1 || name.endsWith(".tif") || name.endsWith(".tiff");
        File target = count > 1 && !name.endsWith(".tif") && !name.endsWith(".tiff")
                ? new File(requested.getParentFile(), requested.getName().replaceFirst("\\.[^.]+$", "") + ".tif")
                : requested;
        if (!tiff) {
            if (!ImageIO.write(plane(labels, axes, 0, 0, false), "png", target))
                throw new IOException("No PNG writer is available.");
            return target;
        }
        Iterator<ImageWriter> writers = ImageIO.getImageWritersByFormatName("TIFF");
        if (!writers.hasNext()) throw new IOException("No TIFF writer is available.");
        ImageWriter writer = writers.next();
        try (ImageOutputStream output = ImageIO.createImageOutputStream(target)) {
            writer.setOutput(output);
            writer.prepareWriteSequence(null);
            for (int t = 0; t < frames; t++) {
                for (int z = 0; z < depth; z++) {
                    if (Thread.currentThread().isInterrupted()) throw new InterruptedIOException("Saving labels cancelled.");
                    BufferedImage image = plane(labels, axes, z, t, true);
                    IIOMetadata metadata = z == 0 && t == 0 ? stackMetadata(writer, image, depth, frames) : null;
                    writer.writeToSequence(new IIOImage(image, null, metadata), null);
                }
            }
            writer.endWriteSequence();
        } finally {
            writer.dispose();
        }
        return target;
    }

    private static IIOMetadata stackMetadata(ImageWriter writer, BufferedImage image, int depth, int frames)
            throws IOException {
        IIOMetadata metadata = writer.getDefaultImageMetadata(new ImageTypeSpecifier(image), writer.getDefaultWriteParam());
        String format = metadata.getNativeMetadataFormatName();
        IIOMetadataNode root = new IIOMetadataNode(format);
        IIOMetadataNode ifd = new IIOMetadataNode("TIFFIFD");
        IIOMetadataNode field = new IIOMetadataNode("TIFFField");
        field.setAttribute("number", "270");
        IIOMetadataNode values = new IIOMetadataNode("TIFFAsciis");
        IIOMetadataNode value = new IIOMetadataNode("TIFFAscii");
        value.setAttribute("value", "ImageJ=1.54\nimages=" + Math.multiplyExact(depth, frames)
                + "\nchannels=1\nslices=" + depth + "\nframes=" + frames + "\nhyperstack=true\nmode=grayscale\n");
        values.appendChild(value);
        field.appendChild(values);
        ifd.appendChild(field);
        root.appendChild(ifd);
        metadata.mergeTree(format, root);
        return metadata;
    }

    private static <T extends RealType<T> & NativeType<T>> BufferedImage plane(
            RandomAccessibleInterval<T> labels, String axes, int z, int t, boolean floating) throws IOException {
        int xAxis = axes.indexOf('x'), yAxis = axes.indexOf('y');
        int width = Math.toIntExact(labels.dimension(xAxis)), height = Math.toIntExact(labels.dimension(yAxis));
        BufferedImage image;
        if (floating) {
            ComponentColorModel colors = new ComponentColorModel(ColorSpace.getInstance(ColorSpace.CS_GRAY),
                    false, false, Transparency.OPAQUE, DataBuffer.TYPE_FLOAT);
            ComponentSampleModel samples = new ComponentSampleModel(DataBuffer.TYPE_FLOAT, width, height, 1, width, new int[] {0});
            WritableRaster raster = Raster.createWritableRaster(samples, new DataBufferFloat(Math.multiplyExact(width, height)), null);
            image = new BufferedImage(colors, raster, false, null);
        } else {
            image = new BufferedImage(width, height, BufferedImage.TYPE_USHORT_GRAY);
        }
        RandomAccess<T> access = labels.randomAccess();
        long[] position = new long[labels.numDimensions()];
        if (axes.indexOf('z') >= 0) position[axes.indexOf('z')] = z;
        int timeAxis = axes.indexOf('t') >= 0 ? axes.indexOf('t') : axes.indexOf('b');
        if (timeAxis >= 0) position[timeAxis] = t;
        for (int y = 0; y < height; y++) {
            position[yAxis] = y;
            for (int x = 0; x < width; x++) {
                position[xAxis] = x;
                access.setPosition(position);
                double value = access.get().getRealDouble();
                if (!floating && value > 65535) throw new IOException("Labels above 65535 require TIFF output.");
                image.getRaster().setSample(x, y, 0, Double.isFinite(value) && value > 0 ? value : 0);
            }
        }
        return image;
    }
}
