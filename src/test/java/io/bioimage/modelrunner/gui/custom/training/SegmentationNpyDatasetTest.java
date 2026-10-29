package io.bioimage.modelrunner.gui.custom.training;

import static org.junit.Assert.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.function.IntToDoubleFunction;

import javax.imageio.ImageIO;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer.Dimensionality;
import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer.Framework;
import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer.PreparedDataset;
import io.bioimage.modelrunner.numpy.NpyImageFile;

public class SegmentationNpyDatasetTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test public void normalizesMixedNumpyRasterPairsUsingLinksOnly() throws Exception {
        File root = temporary.newFolder("mixed");
        File source = new File(root, "case_sample.npy");
        write(source, new int[] {2, 128, 160}, "<f4", false, i -> i % 211);
        BufferedImage mask = new BufferedImage(160, 128, BufferedImage.TYPE_USHORT_GRAY);
        mask.getRaster().setSample(50, 50, 0, 300);
        ImageIO.write(mask, "png", new File(root, "case_seg.png"));
        byte[] before = Files.readAllBytes(source.toPath());
        for (Framework framework : Framework.values()) {
            PreparedDataset dataset = SegmentationDatasetPreparer.prepare(root.getAbsolutePath(), "model",
                    temporary.newFolder("models-" + framework).getAbsolutePath(), 0.15, framework, null);
            assertTrue(dataset.isGenerated());
            assertEquals(Dimensionality.TWO_D, dataset.getDimensionality());
            File linked = new File(dataset.getDatasetRoot(), "train/images/case.npy");
            assertTrue(linked.isFile());
            assertTrue(Files.isSameFile(source.toPath(), linked.toPath()));
            assertTrue(new File(dataset.getDatasetRoot(), "train/masks/case_mask.png").isFile());
        }
        assertArrayEquals(before, Files.readAllBytes(source.toPath()));
        write(source, new int[] {3, 128, 160}, "<f4", false, i -> i % (128 * 160) + i / (128 * 160));
        PreparedDataset grayscale = SegmentationDatasetPreparer.prepare(root.getAbsolutePath(), "gray",
                temporary.getRoot().getAbsolutePath(), 0.15, Framework.STARDIST, null);
        assertEquals("NumPy RGB review must use the same grayscale tolerance as raster inputs", 1, grayscale.getTargetImageChannels());
    }

    @Test public void reviewsVolumesAndRejectsTheFirstInvalidMask() throws Exception {
        File root = temporary.newFolder("volumes");
        File image = new File(root, "images/case.npy"), mask = new File(root, "masks/case_mask.npy");
        write(image, new int[] {128, 160, 1, 4}, "<f4", true, i -> i % 127);
        write(mask, new int[] {128, 160, 4}, ">u4", true, i -> 70000);
        assertEquals(Dimensionality.THREE_D, SegmentationDatasetPreparer.inspectUnetDimensionality(root));
        try (NpyImageFile numpy = new NpyImageFile(mask)) {
            numpy.resolveVolume(128, 160, 4, true);
            assertEquals(70000, numpy.value(12, 40, 3, 0), 0);
        }
        for (Framework framework : Framework.values()) {
            PreparedDataset dataset = SegmentationDatasetPreparer.prepare(root.getAbsolutePath(), "vol",
                    temporary.getRoot().getAbsolutePath(), 0.15, framework, null);
            assertTrue(dataset.is3D());
        }
        write(mask, new int[] {128, 160, 4}, "<f4", false, i -> 0.5);
        File later = new File(root, "images/z-later.npy");
        Files.write(later.toPath(), new byte[] {0});
        Files.write(new File(root, "masks/z-later_mask.npy").toPath(), new byte[] {0});
        try {
            SegmentationDatasetPreparer.prepare(root.getAbsolutePath(), "bad", temporary.getRoot().getAbsolutePath(),
                    0.15, Framework.UNET, null);
            fail("Fractional mask IDs must fail before later sources are inspected");
        } catch (java.io.IOException e) {
            assertTrue(e.getMessage().contains(mask.getAbsolutePath()));
            assertTrue(e.getMessage().contains(NpyImageFile.DOCUMENTATION_URL));
            assertFalse(e.getMessage().contains("z-later"));
        }
    }

    @Test public void ignoresSameStemNumpyImageCachesBeforeReadingHeaders() throws Exception {
        File root = temporary.newFolder("cached");
        File images = new File(root, "images"), masks = new File(root, "masks");
        images.mkdirs();
        masks.mkdirs();
        BufferedImage image = new BufferedImage(160, 128, BufferedImage.TYPE_USHORT_GRAY);
        ImageIO.write(image, "tif", new File(images, "case.tif"));
        ImageIO.write(image, "tif", new File(masks, "case_mask.tif"));
        Files.write(new File(images, "case.npy").toPath(), new byte[] {0});
        write(new File(images, "other.npy"), new int[] {128, 160}, "<f4", false, i -> i % 127);
        ImageIO.write(image, "tif", new File(masks, "other_mask.tif"));
        assertEquals(Dimensionality.TWO_D, SegmentationDatasetPreparer.inspectUnetDimensionality(root));
        for (Framework framework : Framework.values()) {
            java.util.List<String> messages = new java.util.ArrayList<>();
            PreparedDataset dataset = SegmentationDatasetPreparer.prepare(root.getAbsolutePath(), "cache",
                    temporary.getRoot().getAbsolutePath(), 0.15, framework, messages::add);
            assertFalse(dataset.isGenerated());
            assertTrue(messages.stream().anyMatch(message -> message.contains("source_images=2")));
        }
    }

    private static void write(File file, int[] shape, String dtype, boolean fortran, IntToDoubleFunction value) throws Exception {
        Files.createDirectories(file.getParentFile().toPath());
        String tuple = Arrays.toString(shape).replace('[', '(').replace(']', ')');
        String text = "{'descr': '" + dtype + "', 'fortran_order': " + (fortran ? "True" : "False") + ", 'shape': " + tuple + ", }";
        int padding = (64 - (10 + text.length() + 1) % 64) % 64;
        byte[] header = (text + " ".repeat(padding) + "\n").getBytes(StandardCharsets.US_ASCII);
        int count = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        ByteBuffer result = ByteBuffer.allocate(10 + header.length + 4 * count).order(ByteOrder.LITTLE_ENDIAN);
        result.put(new byte[] {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0}).putShort((short) header.length).put(header);
        result.order(dtype.startsWith(">") ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            if (dtype.charAt(1) == 'f') result.putFloat((float) value.applyAsDouble(i));
            else result.putInt((int) value.applyAsDouble(i));
        }
        Files.write(file.toPath(), result.array());
    }
}
