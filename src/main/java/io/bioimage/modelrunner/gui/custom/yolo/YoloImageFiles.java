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
package io.bioimage.modelrunner.gui.custom.yolo;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

public final class YoloImageFiles {

    private static final Set<String> IMAGE_EXTENSIONS = new HashSet<String>(Arrays.asList(
            ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"));

    private YoloImageFiles() {}

    /**
     * Returns whether has supported image extension.
     *
     * @param file the file.
     * @return true if has supported image extension; false otherwise.
     */
    public static boolean hasSupportedImageExtension(File file) {
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

    /**
     * Returns whether can read image.
     *
     * @param file the file.
     * @return true if can read image; false otherwise.
     */
    public static boolean canReadImage(File file) {
        if (!hasSupportedImageExtension(file)) {
            return false;
        }
        ImageInputStream stream = null;
        try {
            stream = ImageIO.createImageInputStream(file);
            if (stream == null) {
                return false;
            }
            Iterator<ImageReader> readers = ImageIO.getImageReaders(stream);
            return readers.hasNext();
        } catch (IOException e) {
            return false;
        } finally {
            if (stream != null) {
                try {
                    stream.close();
                } catch (IOException e) {
                    // Ignore validation cleanup errors.
                }
            }
        }
    }

    /**
     * Returns the result of readable images in directory.
     *
     * @param directory the directory.
     * @return the resulting list.
     */
    public static List<File> readableImagesInDirectory(File directory) {
        if (directory == null || !directory.isDirectory()) {
            return Collections.emptyList();
        }
        File[] children = directory.listFiles(file -> file.isFile() && hasSupportedImageExtension(file));
        if (children == null || children.length == 0) {
            return Collections.emptyList();
        }
        Arrays.sort(children, (a, b) -> a.getName().compareToIgnoreCase(b.getName()));
        List<File> readable = new ArrayList<File>();
        for (File child : children) {
            if (canReadImage(child)) {
                readable.add(child);
            }
        }
        return readable;
    }

    /**
     * Returns the result of preview image in directory.
     *
     * @param directory the directory.
     * @return the resulting file.
     */
    public static File previewImageInDirectory(File directory) {
        List<File> images = readableImagesInDirectory(directory);
        return images.isEmpty() ? null : images.get(images.size() / 2);
    }

    /**
     * Returns whether valid dropped path.
     *
     * @param file the file.
     * @return true if valid dropped path; false otherwise.
     */
    public static boolean isValidDroppedPath(File file) {
        if (file == null) {
            return false;
        }
        if (file.isDirectory()) {
            return previewImageInDirectory(file) != null;
        }
        return canReadImage(file);
    }

    /**
     * Returns the result of supported extensions.
     *
     * @return the resulting set.
     */
    public static Set<String> supportedExtensions() {
        return Collections.unmodifiableSet(IMAGE_EXTENSIONS);
    }
}
