package io.bioimage.modelrunner.gui.custom.crossgoose;

import static org.junit.Assert.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import javax.imageio.ImageIO;

import org.apposed.appose.Service;
import org.apposed.appose.TaskException;
import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer;
import io.bioimage.modelrunner.gui.custom.training.SegmentationDatasetPreparer.Framework;
import io.bioimage.modelrunner.gui.custom.training.ValidationRequests;
import io.bioimage.modelrunner.model.special.crossgoose.CrossGoose;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.integer.UnsignedShortType;

/** Opt in with JDLL_CROSS_GOOSE_INTEGRATION=true; uses the real revision-pinned Pixi worker. */
public class CrossGooseIntegrationTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test(timeout = 300000)
    public void trainsFineTunesReloadsAndInfersThroughTheActualJavaBridge() throws Exception {
        Assume.assumeTrue(Boolean.parseBoolean(System.getenv("JDLL_CROSS_GOOSE_INTEGRATION")));
        File dataset = dataset();
        File output = temporary.newFolder("scratch");
        Files.writeString(new File(output, "training-ui.log").toPath(), "Starting training\n");
        List<String> messages = new ArrayList<>();
        List<String> previews = new ArrayList<>();
        List<Map<String, Object>> lifecycle = new ArrayList<>();
        try (ValidationRequests requests = new ValidationRequests()) {
            Map<String, Object> values = config(dataset, output);
            values.put("_jdll_validation_requests", requests.getDirectory());
            CrossGoose.train(values, progress -> {
                if (progress.getValidationTotalLoss() != null) {
                    assertNotNull(progress.getPrimaryMetric());
                    assertEquals("AP@0.75", progress.getPrimaryMetricName());
                }
            }, preview -> previews.add(preview.getPreviewJsonPath()), messages::add, null, event -> {
                requests.accept(event);
                lifecycle.add(event);
                if ("full_validation".equals(event.get("type")) && "ready".equals(event.get("status"))) {
                    try { assertNotNull(requests.request()); }
                    catch (java.io.IOException failure) { throw new java.io.UncheckedIOException(failure); }
                }
            });
        }
        File best = new File(output, "checkpoints/best.ckpt");
        File last = new File(output, "checkpoints/last.ckpt");
        assertTrue(best.isFile());
        assertTrue(last.isFile());
        assertFalse(new File(output, "scratch").exists());
        assertTrue(CrossGooseModelRegistry.isModelPath(output.getAbsolutePath()));
        assertEquals(1, previews.size());
        assertTrue(new File(previews.get(0)).isFile());
        assertTrue(lifecycle.stream().anyMatch(e -> "checkpoint".equals(e.get("type"))));
        assertTrue(lifecycle.stream().anyMatch(e -> "full_validation".equals(e.get("type"))
                && "completed".equals(e.get("status"))));
        assertTrue(messages.stream().anyMatch(s -> s.startsWith("Resolved Cross-GOOSE model:")));
        CrossGoose model = CrossGoose.fromFile(best.getAbsolutePath(), null, "cpu");
        try {
            for (int i = 0; i < 2; i++) {
                List<Tensor<UnsignedShortType>> result = model.inference(
                        Tensor.build("input", "xycb", ArrayImgs.unsignedBytes(40, 32, 1, 1)));
                assertEquals(1, result.size());
                assertEquals("yx", result.get(0).getAxesOrderString());
                assertEquals(32, result.get(0).getData().dimension(0));
                assertEquals(40, result.get(0).getData().dimension(1));
            }
        } finally {
            model.close();
        }
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(best.toPath()));
        File destination = temporary.newFolder("fine-tuned");
        Map<String, Object> fineTune = config(dataset, destination);
        fineTune.put("starting_point", "fine_tune");
        fineTune.put("base_model", best.getAbsolutePath());
        CrossGoose.train(fineTune, null, null, null, null);
        assertTrue(new File(destination, "checkpoints/last.ckpt").isFile());
        assertArrayEquals(digest, MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(best.toPath())));
    }

    @Test(timeout = 300000)
    public void cancellationAndBackendFailureNeverReportSuccessOrOverwriteSavedWeights() throws Exception {
        Assume.assumeTrue(Boolean.parseBoolean(System.getenv("JDLL_CROSS_GOOSE_INTEGRATION")));
        File output = temporary.newFolder("cancelled");
        Map<String, Object> config = config(dataset(), output);
        config.put("epochs", 3);
        AtomicReference<Service> worker = new AtomicReference<>();
        try {
            CrossGoose.train(config, progress -> {
                if (progress.getValidationTotalLoss() != null) worker.get().kill();
            }, null, null, worker::set);
            fail("Killing the worker must fail the pending task, not report successful completion");
        } catch (TaskException expected) {
            assertTrue(new File(output, "checkpoints/last.ckpt").isFile());
        }
        assertNull(worker.get());
        byte[] checkpoint = Files.readAllBytes(new File(output, "checkpoints/last.ckpt").toPath());
        try {
            CrossGoose.train(config, null, null, null, null);
            fail("An existing run must never be overwritten");
        } catch (TaskException expected) {
            assertTrue(expected.getMessage().contains("not empty"));
        }
        assertArrayEquals(checkpoint, Files.readAllBytes(new File(output, "checkpoints/last.ckpt").toPath()));
    }

    private Map<String, Object> config(File dataset, File output) {
        Map<String, Object> values = new java.util.LinkedHashMap<>();
        values.put("model_name", output.getName());
        values.put("output_dir", output.getParent());
        values.put("model_dir", output.getAbsolutePath());
        values.put("dataset_path", dataset.getAbsolutePath());
        values.put("architecture", "default"); values.put("starting_point", "scratch");
        values.put("device", "cpu"); values.put("epochs", 1);
        values.put("batch_size", 1); values.put("val_batch_size", 1); values.put("num_workers", 0);
        values.put("patch_size", 32); values.put("val_patch_size", 32);
        values.put("n_samples", 12); values.put("n_steps", 2);
        values.put("steps_per_epoch", 2);
        return values;
    }

    private File dataset() throws Exception {
        File root = temporary.newFolder();
        for (String split : new String[] {"train", "val"}) {
            File images = new File(root, split + "/images"), masks = new File(root, split + "/masks");
            assertTrue(images.mkdirs()); assertTrue(masks.mkdirs());
            for (int i = 0; i < ("train".equals(split) ? 2 : 1); i++) {
                BufferedImage image = new BufferedImage(40, 32, BufferedImage.TYPE_BYTE_GRAY);
                BufferedImage mask = new BufferedImage(40, 32, BufferedImage.TYPE_USHORT_GRAY);
                for (int y = 0; y < 32; y++) for (int x = 0; x < 40; x++) {
                    int label = (x - 12) * (x - 12) + (y - 12) * (y - 12) < 49 ? 1 : 0;
                    image.getRaster().setSample(x, y, 0, label * 150 + (x + y + i) % 20);
                    mask.getRaster().setSample(x, y, 0, label);
                }
                ImageIO.write(image, "png", new File(images, "sample" + i + "_raw.png"));
                ImageIO.write(mask, "png", new File(masks, "sample" + i + "_masks.png"));
            }
        }
        return SegmentationDatasetPreparer.prepare(root.getAbsolutePath(), "smoke",
                temporary.getRoot().getAbsolutePath(), .15, Framework.CROSS_GOOSE, null).getDatasetRoot();
    }
}
