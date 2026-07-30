/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer.Framework;
import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer.PreparedDataset;

public class SegmentationDatasetPreparerTest {

    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void reusesBackendCompatibleRootAliases() throws Exception {
        List<String> imageAliases = Arrays.asList("images", "image", "imgs", "img", "data");
        List<String> maskAliases = Arrays.asList("masks", "mask", "labels", "label", "gt");
        for (String imageAlias : imageAliases) {
            for (String maskAlias : maskAliases) {
                File root = temporaryFolder.newFolder("root-" + imageAlias + "-" + maskAlias);
                writePair(new File(root, imageAlias), new File(root, maskAlias), "sample", "sample_mask");

                PreparedDataset prepared = prepare(root, "aliases");

                assertFalse(imageAlias + "/" + maskAlias, prepared.isGenerated());
                assertEquals(root.getCanonicalFile(), prepared.getDatasetRoot().getCanonicalFile());
            }
        }
    }

    @Test
    public void normalizesNamesOutsideBackendSuffixContract() throws Exception {
        List<String> extraImageSuffixes = Arrays.asList(
                "_images", "-images", "_imgs", "-imgs",
                "_sample", "-sample", "_samples", "-samples");
        List<String> extraMaskSuffixes = Arrays.asList(
                "_masks", "-masks", "_seg", "-seg", "_segmentation", "-segmentation");
        int index = 0;
        for (String suffix : extraImageSuffixes) {
            File root = temporaryFolder.newFolder("image-suffix-" + index++);
            writePair(new File(root, "images"), new File(root, "masks"),
                    "sample" + suffix, "sample_mask");

            PreparedDataset prepared = prepare(root, "suffixes");

            assertTrue(prepared.isGenerated());
            assertCanonicalPair(prepared.getDatasetRoot(), "train", "sample");
        }
        for (String suffix : extraMaskSuffixes) {
            File root = temporaryFolder.newFolder("mask-suffix-" + index++);
            writePair(new File(root, "images"), new File(root, "masks"),
                    "sample", "sample" + suffix);

            PreparedDataset prepared = prepare(root, "suffixes");

            assertTrue(prepared.isGenerated());
            assertCanonicalPair(prepared.getDatasetRoot(), "train", "sample");
        }
    }

    @Test
    public void reusesEveryBackendSupportedSuffixCombination() throws Exception {
        List<String> imageSuffixes = Arrays.asList("", "_image", "-image", "_img", "-img", "_raw", "-raw");
        List<String> maskSuffixes = Arrays.asList("", "_mask", "-mask", "_label", "-label",
                "_labels", "-labels", "_gt", "-gt");
        int index = 0;
        for (String imageSuffix : imageSuffixes) {
            for (String maskSuffix : maskSuffixes) {
                File root = temporaryFolder.newFolder("direct-suffix-" + index++);
                writePair(new File(root, "images"), new File(root, "masks"),
                        "sample" + imageSuffix, "sample" + maskSuffix);

                PreparedDataset prepared = prepare(root, "direct-suffixes");

                assertFalse(imageSuffix + "/" + maskSuffix, prepared.isGenerated());
            }
        }
    }

    @Test
    public void supportsAndNormalizesValidationAliases() throws Exception {
        for (String validation : Arrays.asList("val", "valid", "validation")) {
            File root = temporaryFolder.newFolder("split-" + validation);
            writePair(new File(root, "train/images"), new File(root, "train/masks"), "train", "train_mask");
            writePair(new File(root, validation + "/images"), new File(root, validation + "/masks"),
                    "validation", "validation_mask");

            PreparedDataset prepared = prepare(root, "validation");

            if ("val".equals(validation)) {
                assertFalse(prepared.isGenerated());
            } else {
                assertTrue(prepared.isGenerated());
                assertCanonicalPair(prepared.getDatasetRoot(), "train", "train");
                assertCanonicalPair(prepared.getDatasetRoot(), "val", "validation");
            }
        }
    }

    @Test
    public void supportsImagesFirstSplitLayoutThroughLinks() throws Exception {
        File root = temporaryFolder.newFolder("images-first");
        writePair(new File(root, "images/train"), new File(root, "masks/train"), "sample", "sample_mask");

        PreparedDataset prepared = prepare(root, "images-first");

        assertTrue(prepared.isGenerated());
        assertCanonicalPair(prepared.getDatasetRoot(), "train", "sample");
    }

    @Test
    public void supportsRootAndSplitMixedFoldersWhenMasksHaveSuffixes() throws Exception {
        File rootMixed = temporaryFolder.newFolder("root-mixed");
        writeImage(rootMixed.toPath().resolve("sample.png"), 8, 8, 0);
        writeImage(rootMixed.toPath().resolve("sample_mask.png"), 8, 8, 1);
        PreparedDataset rootPrepared = prepare(rootMixed, "root-mixed");
        assertTrue(rootPrepared.isGenerated());
        assertCanonicalPair(rootPrepared.getDatasetRoot(), "train", "sample");

        File splitMixed = temporaryFolder.newFolder("split-mixed");
        writeImage(splitMixed.toPath().resolve("train/sample.png"), 8, 8, 0);
        writeImage(splitMixed.toPath().resolve("train/sample_label.png"), 8, 8, 1);
        writeImage(splitMixed.toPath().resolve("validation/other.png"), 8, 8, 0);
        writeImage(splitMixed.toPath().resolve("validation/other_seg.png"), 8, 8, 1);
        PreparedDataset splitPrepared = prepare(splitMixed, "split-mixed");
        assertTrue(splitPrepared.isGenerated());
        assertCanonicalPair(splitPrepared.getDatasetRoot(), "train", "sample");
        assertCanonicalPair(splitPrepared.getDatasetRoot(), "val", "other");
    }

    @Test
    public void rejectsMixedFoldersWithoutMaskSuffix() throws Exception {
        File root = temporaryFolder.newFolder("ambiguous-mixed");
        writeImage(root.toPath().resolve("sample.tif.png"), 8, 8, 0);
        writeImage(root.toPath().resolve("sample.png"), 8, 8, 1);

        try {
            prepare(root, "ambiguous");
            fail("A mixed folder without a mask suffix must not be accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("Could not find image/mask pairs")
                    || expected.getMessage().contains("Missing masks"));
        }
    }

    @Test
    public void rejectsMissingMasksAndMismatchedShapes() throws Exception {
        File missing = temporaryFolder.newFolder("missing");
        writeImage(missing.toPath().resolve("images/sample.png"), 8, 8, 0);
        Files.createDirectories(missing.toPath().resolve("masks"));
        try {
            prepare(missing, "missing");
            fail("Missing masks must fail");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("Could not find image/mask pairs")
                    || expected.getMessage().contains("Missing masks"));
        }

        File mismatch = temporaryFolder.newFolder("mismatch");
        writePair(new File(mismatch, "images"), new File(mismatch, "masks"),
                "sample", "sample_mask", 8, 9);
        try {
            prepare(mismatch, "mismatch");
            fail("Shape mismatches must fail when no valid pair remains");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("No valid training image/mask pairs"));
        }
    }

    @Test
    public void selectsStarDistChannelsFromImageContent() throws Exception {
        File grayscale = temporaryFolder.newFolder("stardist-grayscale");
        writePair(new File(grayscale, "images"), new File(grayscale, "masks"), "gray", "gray_mask");
        PreparedDataset grayscalePrepared = prepareStarDist(grayscale, "gray");
        assertEquals(1, grayscalePrepared.getTargetImageChannels());
        assertEquals("grayscale", grayscalePrepared.getImageChannels());
        assertFalse(grayscalePrepared.isGenerated());

        File encodedGray = temporaryFolder.newFolder("stardist-rgb-gray");
        writeRgbPair(encodedGray, "gray", true);
        PreparedDataset encodedGrayPrepared = prepareStarDist(encodedGray, "encoded-gray");
        assertEquals(1, encodedGrayPrepared.getTargetImageChannels());
        assertTrue(encodedGrayPrepared.isGenerated());

        File rgb = temporaryFolder.newFolder("stardist-rgb");
        writeRgbPair(rgb, "rgb", false);
        PreparedDataset rgbPrepared = prepareStarDist(rgb, "rgb");
        assertEquals(3, rgbPrepared.getTargetImageChannels());
        assertEquals("rgb", rgbPrepared.getImageChannels());
        assertFalse(rgbPrepared.isGenerated());

        File twoChannel = temporaryFolder.newFolder("stardist-two-channel");
        writeTwoChannelPair(twoChannel, "two");
        PreparedDataset twoChannelPrepared = prepareStarDist(twoChannel, "two");
        assertEquals(3, twoChannelPrepared.getTargetImageChannels());
        assertTrue(twoChannelPrepared.isGenerated());
    }

    @Test
    public void createsChannelStratifiedSplitForMixedStarDistDataset() throws Exception {
        File root = temporaryFolder.newFolder("stardist-mixed");
        for (int i = 0; i < 4; i++) {
            writePair(new File(root, "images"), new File(root, "masks"),
                    "gray" + i, "gray" + i + "_mask");
            writeRgbPair(root, "rgb" + i, false);
        }

        PreparedDataset prepared = prepareStarDist(root, "mixed");

        assertEquals(3, prepared.getTargetImageChannels());
        assertTrue(prepared.isGenerated());
        assertEquals(2, countFiles(new File(prepared.getDatasetRoot(), "val/images")));
        assertEquals(6, countFiles(new File(prepared.getDatasetRoot(), "train/images")));
    }

    @Test
    public void detectsVolumesAndRejectsMixedDimensionality() throws Exception {
        File volumeRoot = temporaryFolder.newFolder("stardist-volume");
        for (int i = 0; i < 3; i++) {
            writeVolume(new File(volumeRoot, "images/volume" + i + ".tif").toPath(), 8, 7, 4, i);
            writeVolume(new File(volumeRoot, "masks/volume" + i + "_mask.tif").toPath(), 8, 7, 4, 1);
        }
        PreparedDataset volume = prepareStarDist(volumeRoot, "volume");
        assertTrue(volume.is3D());
        assertEquals(3, volume.getDimensionality().getDimensions());

        File mixedRoot = temporaryFolder.newFolder("stardist-mixed-dimensionality");
        writeVolume(new File(mixedRoot, "images/volume.tif").toPath(), 8, 7, 3, 0);
        writeVolume(new File(mixedRoot, "masks/volume_mask.tif").toPath(), 8, 7, 3, 1);
        writeImage(new File(mixedRoot, "images/plane.png").toPath(), 8, 7, 0);
        writeImage(new File(mixedRoot, "masks/plane_mask.png").toPath(), 8, 7, 1);
        try {
            prepareStarDist(mixedRoot, "mixed-dimensionality");
            fail("Mixed 2D and 3D datasets must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("mix 2D images and 3D volumes"));
        }
    }

    private PreparedDataset prepare(File root, String modelName) throws IOException {
        File models = new File(temporaryFolder.getRoot(), "models/unet");
        return SegmentationDatasetPreparer.prepare(root.getAbsolutePath(), modelName, models.getAbsolutePath(),
                0.15d, Framework.UNET, null);
    }

    private PreparedDataset prepareStarDist(File root, String modelName) throws IOException {
        File models = new File(temporaryFolder.getRoot(), "models/stardist");
        return SegmentationDatasetPreparer.prepare(root.getAbsolutePath(), modelName, models.getAbsolutePath(),
                0.15d, Framework.STARDIST, null);
    }

    private static void writePair(File imageDir, File maskDir, String imageStem, String maskStem)
            throws IOException {
        writePair(imageDir, maskDir, imageStem, maskStem, 8, 8);
    }

    private static void writePair(File imageDir, File maskDir, String imageStem, String maskStem,
            int imageWidth, int maskWidth) throws IOException {
        writeImage(imageDir.toPath().resolve(imageStem + ".png"), imageWidth, 8, 0);
        writeImage(maskDir.toPath().resolve(maskStem + ".png"), maskWidth, 8, 1);
    }

    private static void writeImage(Path path, int width, int height, int value) throws IOException {
        Files.createDirectories(path.getParent());
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_USHORT_GRAY);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image.getRaster().setSample(x, y, 0, value);
            }
        }
        if (!ImageIO.write(image, "png", path.toFile())) {
            throw new IOException("Could not write test PNG: " + path);
        }
    }

    private static void writeVolume(Path path, int width, int height, int depth, int value) throws IOException {
        Files.createDirectories(path.getParent());
        Iterator<ImageWriter> writers = ImageIO.getImageWritersByFormatName("TIFF");
        if (!writers.hasNext()) {
            throw new IOException("No TIFF writer available for tests.");
        }
        ImageWriter writer = writers.next();
        try (ImageOutputStream output = ImageIO.createImageOutputStream(path.toFile())) {
            writer.setOutput(output);
            writer.prepareWriteSequence(null);
            for (int z = 0; z < depth; z++) {
                BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_USHORT_GRAY);
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        image.getRaster().setSample(x, y, 0, value + z);
                    }
                }
                writer.writeToSequence(new IIOImage(image, null, null), null);
            }
            writer.endWriteSequence();
        } finally {
            writer.dispose();
        }
    }

    private static void writeRgbPair(File root, String stem, boolean grayscale) throws IOException {
        Path imagePath = new File(root, "images/" + stem + ".png").toPath();
        Files.createDirectories(imagePath.getParent());
        BufferedImage image = new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int red = 20 + x;
                int green = grayscale ? red : 40 + y;
                int blue = grayscale ? red : 80 + x + y;
                image.setRGB(x, y, (red << 16) | (green << 8) | blue);
            }
        }
        if (!ImageIO.write(image, "png", imagePath.toFile())) {
            throw new IOException("Could not write test RGB PNG: " + imagePath);
        }
        writeImage(new File(root, "masks/" + stem + "_mask.png").toPath(), 8, 8, 1);
    }

    private static void writeTwoChannelPair(File root, String stem) throws IOException {
        Path imagePath = new File(root, "images/" + stem + ".png").toPath();
        Files.createDirectories(imagePath.getParent());
        ComponentColorModel colorModel = new ComponentColorModel(
                ColorSpace.getInstance(ColorSpace.CS_GRAY), new int[] {8, 8},
                true, false, Transparency.TRANSLUCENT, DataBuffer.TYPE_BYTE);
        WritableRaster raster = colorModel.createCompatibleWritableRaster(8, 8);
        BufferedImage image = new BufferedImage(colorModel, raster, false, null);
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                raster.setSample(x, y, 0, x + y);
                raster.setSample(x, y, 1, x * 2 + y);
            }
        }
        if (!ImageIO.write(image, "png", imagePath.toFile())) {
            throw new IOException("Could not write test two-channel PNG: " + imagePath);
        }
        writeImage(new File(root, "masks/" + stem + "_mask.png").toPath(), 8, 8, 1);
    }

    private static int countFiles(File folder) {
        File[] files = folder.listFiles(File::isFile);
        return files == null ? 0 : files.length;
    }

    private static void assertCanonicalPair(File root, String split, String stem) {
        File imageDir = new File(new File(root, split), "images");
        File maskDir = new File(new File(root, split), "masks");
        assertTrue(new File(imageDir, stem + ".png").isFile());
        assertTrue(new File(maskDir, stem + "_mask.png").isFile());
    }
}
