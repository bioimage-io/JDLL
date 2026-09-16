/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.gui.custom.unet;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Stream;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer;

public final class UnetDatasetInspector {

    public enum Dimensionality {
        UNKNOWN,
        TWO_D,
        THREE_D,
        MIXED;

        public boolean hasVolumes() {
            return this == THREE_D || this == MIXED;
        }

        public boolean allows2D() {
            return this != THREE_D;
        }
    }

    private static final int MAX_FILES_TO_REVIEW = 48;
    private static final int MAX_SCAN_DEPTH = 6;
    private static final Set<String> IMAGE_EXTENSIONS = new HashSet<String>(Arrays.asList(
            ".tif", ".tiff", ".png", ".jpg", ".jpeg"));

    private UnetDatasetInspector() {}

    /** Reviews paired headers using the same layouts and pairing rules as UNet training. */
    public static Dimensionality inspectPairedDataset(File path) {
        if (path == null || !path.isDirectory()) {
            return Dimensionality.UNKNOWN;
        }
        try {
            return Dimensionality.valueOf(SegmentationDatasetPreparer.inspectUnetDimensionality(path).name());
        } catch (IOException | RuntimeException e) {
            return Dimensionality.UNKNOWN;
        }
    }

    /**
     * Inspects a dataset path and returns whether it looks 2D or volumetric.
     *
     * @param path the dataset path.
     * @return the inferred dimensionality.
     */
    public static Dimensionality inspect(File path) {
        if (path == null || !path.exists()) {
            return Dimensionality.UNKNOWN;
        }
        List<File> images = imageFiles(path);
        if (images.isEmpty()) {
            return Dimensionality.UNKNOWN;
        }
        for (File image : images) {
            if (isMultipageImage(image)) {
                return Dimensionality.THREE_D;
            }
        }
        return Dimensionality.TWO_D;
    }

    private static List<File> imageFiles(File path) {
        if (path.isFile()) {
            return hasImageExtension(path) ? Arrays.asList(path) : new ArrayList<File>();
        }
        List<File> images = new ArrayList<File>();
        try (Stream<Path> stream = Files.walk(path.toPath(), MAX_SCAN_DEPTH)) {
            stream.filter(Files::isRegularFile)
                    .map(Path::toFile)
                    .filter(UnetDatasetInspector::hasImageExtension)
                    .limit(MAX_FILES_TO_REVIEW)
                    .forEach(images::add);
        } catch (IOException e) {
            return new ArrayList<File>();
        }
        return images;
    }

    private static boolean hasImageExtension(File file) {
        if (file == null || !file.isFile()) {
            return false;
        }
        String name = file.getName().toLowerCase(Locale.ROOT);
        for (String extension : IMAGE_EXTENSIONS) {
            if (name.endsWith(extension)) {
                return true;
            }
        }
        return false;
    }

    private static boolean isMultipageImage(File file) {
        try (ImageInputStream input = ImageIO.createImageInputStream(file)) {
            if (input == null) {
                return hasMultipleTiffDirectories(file);
            }
            Iterator<ImageReader> readers = ImageIO.getImageReaders(input);
            if (!readers.hasNext()) {
                return hasMultipleTiffDirectories(file);
            }
            ImageReader reader = readers.next();
            try {
                reader.setInput(input);
                return reader.getNumImages(true) > 1;
            } finally {
                reader.dispose();
            }
        } catch (IOException | RuntimeException e) {
            return hasMultipleTiffDirectories(file);
        }
    }

    private static boolean hasMultipleTiffDirectories(File file) {
        String name = file == null ? "" : file.getName().toLowerCase(Locale.ROOT);
        if (!name.endsWith(".tif") && !name.endsWith(".tiff")) {
            return false;
        }
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            if (raf.length() < 8) {
                return false;
            }
            boolean littleEndian;
            int b0 = raf.readUnsignedByte();
            int b1 = raf.readUnsignedByte();
            if (b0 == 'I' && b1 == 'I') {
                littleEndian = true;
            } else if (b0 == 'M' && b1 == 'M') {
                littleEndian = false;
            } else {
                return false;
            }
            int magic = readUnsignedShort(raf, littleEndian);
            if (magic == 42) {
                return standardTiffDirectoryCount(raf, littleEndian) > 1;
            }
            if (magic == 43 && raf.length() >= 16) {
                return bigTiffDirectoryCount(raf, littleEndian) > 1;
            }
            return false;
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    private static int standardTiffDirectoryCount(RandomAccessFile raf, boolean littleEndian)
            throws IOException {
        long offset = readUnsignedInt(raf, littleEndian);
        int count = 0;
        while (offset > 0 && offset < raf.length() - 2 && count < 2) {
            raf.seek(offset);
            int entries = readUnsignedShort(raf, littleEndian);
            long nextOffsetPos = offset + 2L + entries * 12L;
            if (nextOffsetPos < 0 || nextOffsetPos > raf.length() - 4) {
                break;
            }
            count++;
            raf.seek(nextOffsetPos);
            offset = readUnsignedInt(raf, littleEndian);
        }
        return count;
    }

    private static int bigTiffDirectoryCount(RandomAccessFile raf, boolean littleEndian)
            throws IOException {
        int offsetSize = readUnsignedShort(raf, littleEndian);
        readUnsignedShort(raf, littleEndian);
        if (offsetSize != 8) {
            return 0;
        }
        long offset = readLong(raf, littleEndian);
        int count = 0;
        while (offset > 0 && offset < raf.length() - 8 && count < 2) {
            raf.seek(offset);
            long entries = readLong(raf, littleEndian);
            long nextOffsetPos = offset + 8L + entries * 20L;
            if (nextOffsetPos < 0 || nextOffsetPos > raf.length() - 8) {
                break;
            }
            count++;
            raf.seek(nextOffsetPos);
            offset = readLong(raf, littleEndian);
        }
        return count;
    }

    private static int readUnsignedShort(RandomAccessFile raf, boolean littleEndian)
            throws IOException {
        int b0 = raf.readUnsignedByte();
        int b1 = raf.readUnsignedByte();
        return littleEndian ? b0 | (b1 << 8) : (b0 << 8) | b1;
    }

    private static long readUnsignedInt(RandomAccessFile raf, boolean littleEndian)
            throws IOException {
        long b0 = raf.readUnsignedByte();
        long b1 = raf.readUnsignedByte();
        long b2 = raf.readUnsignedByte();
        long b3 = raf.readUnsignedByte();
        return littleEndian
                ? b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                : (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
    }

    private static long readLong(RandomAccessFile raf, boolean littleEndian)
            throws IOException {
        long value = 0L;
        for (int i = 0; i < 8; i++) {
            long b = raf.readUnsignedByte();
            value = littleEndian ? value | (b << (8 * i)) : (value << 8) | b;
        }
        return value;
    }
}
