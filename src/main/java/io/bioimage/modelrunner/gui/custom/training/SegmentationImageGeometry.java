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
package io.bioimage.modelrunner.gui.custom.training;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.plugins.tiff.BaselineTIFFTagSet;
import javax.imageio.plugins.tiff.TIFFDirectory;
import javax.imageio.plugins.tiff.TIFFField;
import javax.imageio.stream.ImageInputStream;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/** Lightweight spatial review; ambiguous TIFF axes are left to the Python backend. */
final class SegmentationImageGeometry {
    private static final ObjectMapper JSON = new ObjectMapper();
    private int width;
    private int height;
    private final int pages;
    private int depth;
    private boolean explicit;

    private SegmentationImageGeometry(int width, int height, int pages) {
        this.width = width;
        this.height = height;
        this.pages = pages;
        depth = pages;
        explicit = pages == 1;
    }

    static SegmentationImageGeometry read(File file) throws IOException {
        try (ImageInputStream input = ImageIO.createImageInputStream(file)) {
            if (input == null) {
                throw new IOException("Cannot read image header: " + file);
            }
            Iterator<ImageReader> readers = ImageIO.getImageReaders(input);
            if (!readers.hasNext()) {
                throw new IOException("Unsupported image header: " + file);
            }
            ImageReader reader = readers.next();
            try {
                reader.setInput(input);
                SegmentationImageGeometry geometry = new SegmentationImageGeometry(
                        reader.getWidth(0), reader.getHeight(0), Math.max(1, reader.getNumImages(true)));
                if (reader.getFormatName().toLowerCase(java.util.Locale.ROOT).contains("tif")) {
                    try {
                        TIFFField description = TIFFDirectory.createFromMetadata(reader.getImageMetadata(0))
                                .getTIFFField(BaselineTIFFTagSet.TAG_IMAGE_DESCRIPTION);
                        if (description != null) {
                            geometry.readDescription(description.getAsString(0).trim());
                        }
                    } catch (IOException | RuntimeException e) {
                        // Python performs the definitive metadata validation and reports the reason.
                        geometry.depth = 0;
                    }
                }
                return geometry;
            } finally {
                reader.dispose();
            }
        }
    }

    /** Returns spatial depth, zero if unresolved, or -1 for a confirmed mismatch. */
    static int pairedDepth(SegmentationImageGeometry image, SegmentationImageGeometry mask) {
        if (image.width != mask.width || image.height != mask.height) {
            return -1;
        }
        if (image.depth == 0 || mask.depth == 0) {
            return 0;
        }
        if (image.explicit && mask.explicit) {
            return image.depth == mask.depth ? image.depth : -1;
        }
        // Matching scalar stacks can be treated as volumes. Unequal page counts
        // alone cannot reject a pair: one stack could store channels as pages.
        if (image.depth == mask.depth) {
            return image.depth;
        }
        return 0;
    }

    private void readDescription(String description) throws IOException {
        if (description.startsWith("{")) {
            JsonNode metadata = JSON.readTree(description);
            JsonNode shape = metadata.get("shape");
            String axes = metadata.path("axes").asText("").toUpperCase(java.util.Locale.ROOT);
            if (!axes.isEmpty()) {
                depth = 0;
            }
            if (!axes.isEmpty() && shape != null && shape.isArray() && shape.size() == axes.length()) {
                if (axes.indexOf('X') < 0 || axes.indexOf('Y') < 0) {
                    return;
                }
                width = shape.get(axes.indexOf('X')).asInt();
                height = shape.get(axes.indexOf('Y')).asInt();
                depth = 1;
                for (int i = 0; i < axes.length(); i++) {
                    char axis = axes.charAt(i);
                    int size = shape.get(i).asInt();
                    if (size < 1 || ("CZYXS".indexOf(axis) < 0 && size > 1)) {
                        depth = 0;
                        break;
                    }
                    if (axis == 'Z') {
                        depth = size;
                    }
                }
                explicit = true;
            }
        } else if (description.startsWith("ImageJ=")) {
            int channels = imageJSize(description, "channels");
            int slices = imageJSize(description, "slices");
            int frames = imageJSize(description, "frames");
            depth = frames > 1 ? 0 : slices;
            // An unqualified ImageJ stack is spatial, not necessarily a hyperstack.
            if (channels == 1 && slices == 1 && frames == 1) {
                depth = pages;
            }
            explicit = true;
        } else if (description.contains("<OME") || description.contains(":OME")) {
            int z = omeSize(description, "Z");
            int t = omeSize(description, "T");
            depth = t > 1 ? 0 : z;
            explicit = true;
        }
    }

    private static int imageJSize(String description, String key) {
        Matcher matcher = Pattern.compile("(?m)^" + key + "=(\\d+)\\s*$").matcher(description);
        return matcher.find() ? Integer.parseInt(matcher.group(1)) : 1;
    }

    private static int omeSize(String description, String axis) {
        Matcher matcher = Pattern.compile("\\bSize" + axis + "\\s*=\\s*['\"](\\d+)['\"]")
                .matcher(description);
        if (!matcher.find()) {
            return 0;
        }
        int size = Integer.parseInt(matcher.group(1));
        return matcher.find() ? 0 : size;
    }
}
