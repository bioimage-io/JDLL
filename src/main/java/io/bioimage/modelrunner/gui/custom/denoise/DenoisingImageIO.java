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

import java.awt.image.BufferedImage;
import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageInputStream;
import javax.imageio.stream.ImageOutputStream;

import com.google.gson.GsonBuilder;

import io.bioimage.modelrunner.gui.custom.yolo.YoloImageFiles;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.real.FloatType;

/** Minimal dependency-free filesystem image bridge for denoising. */
public final class DenoisingImageIO {
    private DenoisingImageIO() {}

    public static ImageData read(File file) throws IOException {
        try (ImageInputStream stream = ImageIO.createImageInputStream(file)) {
            if (stream == null) throw new IOException("Cannot open image: " + file);
            Iterator<ImageReader> readers = ImageIO.getImageReaders(stream);
            if (!readers.hasNext()) throw new IOException("No ImageIO reader for: " + file);
            ImageReader reader = readers.next();
            try {
                reader.setInput(stream, false, true);
                int pages = imageCount(reader);
                BufferedImage first = reader.read(0);
                int width = first.getWidth();
                int height = first.getHeight();
                int channels = first.getRaster().getNumBands();
                int bits = maximumSampleSize(first);
                long[] dimensions = pages > 1
                        ? new long[] {width, height, channels, pages}
                        : channels > 1 ? new long[] {width, height, channels} : new long[] {width, height};
                Img<FloatType> result = ArrayImgs.floats(dimensions);
                copyPage(first, result, 0, channels, pages);
                for (int z = 1; z < pages; z++) {
                    BufferedImage page = reader.read(z);
                    if (page.getWidth() != width || page.getHeight() != height
                            || page.getRaster().getNumBands() != channels) {
                        throw new IOException("Image stack pages have inconsistent dimensions or channels: " + file);
                    }
                    copyPage(page, result, z, channels, pages);
                }
                String axes = pages > 1 ? "xycz" : channels > 1 ? "xyc" : "xy";
                return new ImageData(result, axes, bits, channels, pages, extension(file));
            } finally {
                reader.dispose();
            }
        }
    }

    public static File outputFile(File source) {
        String extension = extensionWithDot(source);
        String name = source.getName().substring(0, source.getName().length() - extension.length());
        return available(new File(source.getParentFile(), name + "_denoised" + extension), false);
    }

    public static File outputDirectory(File source) {
        return available(new File(source.getParentFile(), source.getName() + "_denoised"), true);
    }

    private static File available(File preferred, boolean directory) {
        if (!preferred.exists()) return preferred;
        String name = preferred.getName();
        String extension = directory ? "" : extensionWithDot(preferred);
        String stem = extension.isEmpty() ? name : name.substring(0, name.length() - extension.length());
        for (int index = 1; ; index++) {
            File candidate = new File(preferred.getParentFile(), stem + "-" + index + extension);
            if (!candidate.exists()) return candidate;
        }
    }

    public static void write(File output, RandomAccessibleInterval<? extends FloatType> image,
            ImageData source) throws IOException {
        if (source.channels > 4) {
            throw new IOException("Dependency-free ImageIO output cannot safely preserve "
                    + source.channels + " channels. Open the image in the host application instead.");
        }
        if (source.bits > 16) {
            throw new IOException("Dependency-free ImageIO output cannot safely preserve "
                    + source.bits + "-bit samples. Open the image in the host application instead.");
        }
        File parent = output.getParentFile();
        if (parent != null) Files.createDirectories(parent.toPath());
        String format = source.format;
        File temporary = new File(parent, "." + output.getName() + ".tmp");
        try {
            writeImage(temporary, format, image, source);
            atomicMove(temporary, output);
        } finally {
            Files.deleteIfExists(temporary.toPath());
        }
    }

    public static File writeSidecar(File output, File source, double strength,
            Map<String, Object> config, Map<String, Object> backendMetadata) throws IOException {
        return writeSidecar(output, output, source, strength, config, backendMetadata);
    }

    public static File writeSidecar(File output, File reportedOutput, File source, double strength,
            Map<String, Object> config, Map<String, Object> backendMetadata) throws IOException {
        String extension = extensionWithDot(output);
        String stem = extension.isEmpty() ? output.getName()
                : output.getName().substring(0, output.getName().length() - extension.length());
        File sidecar = new File(output.getParentFile(), stem + ".json");
        File temporary = new File(sidecar.getParentFile(), "." + sidecar.getName() + ".tmp");
        Map<String, Object> metadata = new LinkedHashMap<String, Object>();
        metadata.put("source", source.getAbsolutePath());
        metadata.put("output", reportedOutput.getAbsolutePath());
        metadata.put("created_at", Instant.now().toString());
        metadata.put("strength", strength);
        metadata.put("config", config);
        metadata.put("backend", backendMetadata);
        try {
            Files.write(temporary.toPath(), new GsonBuilder().setPrettyPrinting().create()
                    .toJson(metadata).getBytes(java.nio.charset.StandardCharsets.UTF_8));
            atomicMove(temporary, sidecar);
        } finally {
            Files.deleteIfExists(temporary.toPath());
        }
        return sidecar;
    }

    public static List<File> inputFiles(File source) {
        if (source == null) return java.util.Collections.emptyList();
        return source.isDirectory() ? YoloImageFiles.readableImagesInDirectory(source)
                : YoloImageFiles.canReadImage(source) ? java.util.Collections.singletonList(source)
                : java.util.Collections.emptyList();
    }

    public static File temporaryDirectoryFor(File outputDirectory) {
        return new File(outputDirectory.getParentFile(), "." + outputDirectory.getName()
                + ".tmp-" + java.util.UUID.randomUUID().toString());
    }

    public static void publishDirectory(File temporary, File output) throws IOException {
        atomicMove(temporary, output);
    }

    public static void deleteRecursively(File path) {
        if (path == null || !path.exists()) return;
        File[] children = path.listFiles();
        if (children != null) {
            for (File child : children) deleteRecursively(child);
        }
        try { Files.deleteIfExists(path.toPath()); }
        catch (IOException ignored) { /* Best-effort temporary cleanup. */ }
    }

    private static void writeImage(File temporary, String format,
            RandomAccessibleInterval<? extends FloatType> image, ImageData source) throws IOException {
        Iterator<ImageWriter> writers = ImageIO.getImageWritersByFormatName(format);
        if (!writers.hasNext()) throw new IOException("No ImageIO writer for format: " + format);
        ImageWriter writer = writers.next();
        try (ImageOutputStream stream = ImageIO.createImageOutputStream(temporary)) {
            writer.setOutput(stream);
            if (source.pages > 1) {
                if (!writer.canWriteSequence()) throw new IOException("The " + format + " writer cannot write stacks.");
                writer.prepareWriteSequence(null);
                for (int z = 0; z < source.pages; z++) {
                    writer.writeToSequence(new IIOImage(toBufferedImage(image, source, z), null, null), null);
                }
                writer.endWriteSequence();
            } else {
                writer.write(new IIOImage(toBufferedImage(image, source, 0), null, null));
            }
        } finally {
            writer.dispose();
        }
    }

    private static BufferedImage toBufferedImage(RandomAccessibleInterval<? extends FloatType> image,
            ImageData source, int z) {
        int width = (int) image.dimension(0);
        int height = (int) image.dimension(1);
        BufferedImage buffered = createBufferedImage(width, height, source.channels, source.bits);
        WritableRaster raster = buffered.getRaster();
        RandomAccess<? extends FloatType> access = image.randomAccess();
        double max = source.bits >= 31 ? Integer.MAX_VALUE : (1L << source.bits) - 1L;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                access.setPosition(x, 0); access.setPosition(y, 1);
                if (source.channels > 1) access.setPosition(0, 2);
                if (source.pages > 1) access.setPosition(z, 3);
                if (source.channels == 1) {
                    raster.setSample(x, y, 0, clamp(access.get().getRealDouble(), max));
                } else {
                    for (int c = 0; c < source.channels; c++) {
                        access.setPosition(c, 2);
                        raster.setSample(x, y, c, clamp(access.get().getRealDouble(), max));
                    }
                }
            }
        }
        return buffered;
    }

    private static BufferedImage createBufferedImage(int width, int height, int channels, int bits) {
        if (channels == 1) {
            return new BufferedImage(width, height, bits > 8
                    ? BufferedImage.TYPE_USHORT_GRAY : BufferedImage.TYPE_BYTE_GRAY);
        }
        boolean alpha = channels == 2 || channels == 4;
        ColorSpace colorSpace = ColorSpace.getInstance(channels <= 2
                ? ColorSpace.CS_GRAY : ColorSpace.CS_sRGB);
        int transferType = bits > 8 ? DataBuffer.TYPE_USHORT : DataBuffer.TYPE_BYTE;
        int storedBits = bits > 8 ? 16 : 8;
        int[] componentBits = new int[channels];
        java.util.Arrays.fill(componentBits, storedBits);
        ComponentColorModel colorModel = new ComponentColorModel(colorSpace, componentBits,
                alpha, false, alpha ? Transparency.TRANSLUCENT : Transparency.OPAQUE, transferType);
        WritableRaster raster = Raster.createInterleavedRaster(
                transferType, width, height, channels, null);
        return new BufferedImage(colorModel, raster, false, null);
    }

    private static int clamp(double value, double maximum) {
        return (int) Math.round(Math.max(0.0d, Math.min(maximum, value)));
    }

    private static void copyPage(BufferedImage page, Img<FloatType> target, int z, int channels, int pages) {
        Raster raster = page.getRaster();
        RandomAccess<FloatType> access = target.randomAccess();
        for (int y = 0; y < page.getHeight(); y++) {
            for (int x = 0; x < page.getWidth(); x++) {
                access.setPosition(x, 0); access.setPosition(y, 1);
                if (pages > 1) access.setPosition(z, 3);
                for (int c = 0; c < channels; c++) {
                    if (channels > 1) access.setPosition(c, 2);
                    access.get().setReal(raster.getSampleDouble(x, y, c));
                }
            }
        }
    }

    private static int imageCount(ImageReader reader) {
        try { return Math.max(1, reader.getNumImages(true)); }
        catch (IOException e) { return 1; }
    }

    private static int maximumSampleSize(BufferedImage image) {
        int maximum = 8;
        for (int size : image.getSampleModel().getSampleSize()) maximum = Math.max(maximum, size);
        return maximum;
    }

    private static String extension(File file) {
        String extension = extensionWithDot(file);
        if (extension.isEmpty()) return "png";
        String value = extension.substring(1).toLowerCase(Locale.ROOT);
        if ("jpg".equals(value)) return "jpeg";
        if ("tif".equals(value)) return "tiff";
        return value;
    }

    private static String extensionWithDot(File file) {
        String name = file.getName();
        int dot = name.lastIndexOf('.');
        return dot < 0 ? "" : name.substring(dot);
    }

    private static void atomicMove(File source, File target) throws IOException {
        try {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(source.toPath(), target.toPath());
        }
    }

    public static final class ImageData {
        private final RandomAccessibleInterval<FloatType> image;
        private final String axes;
        private final int bits;
        private final int channels;
        private final int pages;
        private final String format;

        private ImageData(RandomAccessibleInterval<FloatType> image, String axes, int bits,
                int channels, int pages, String format) {
            this.image = image; this.axes = axes; this.bits = bits;
            this.channels = channels; this.pages = pages; this.format = format;
        }

        public RandomAccessibleInterval<FloatType> getImage() { return image; }
        public String getAxes() { return axes; }
        public int getChannels() { return channels; }
        public int getPages() { return pages; }
        public int getBits() { return bits; }
    }
}
