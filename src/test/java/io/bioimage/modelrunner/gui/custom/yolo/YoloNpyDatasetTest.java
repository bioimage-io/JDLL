package io.bioimage.modelrunner.gui.custom.yolo;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Map;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import com.google.gson.Gson;

import io.bioimage.modelrunner.numpy.NpyImageFile;

public class YoloNpyDatasetTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test public void preparesReferencedNumpyImagesAndUint16MasksWithoutCopies() throws Exception {
        File root = temporary.newFolder("source");
        for (String split : new String[] {"train", "val"}) {
            File image = new File(root, split + "/images/sample.npy");
            File mask = new File(root, split + "/labels/sample_masks.npy");
            byte[] pixels = new byte[3 * 128 * 128 * 4];
            ByteBuffer floats = ByteBuffer.wrap(pixels).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < pixels.length / 4; i++) floats.putFloat(i % 256);
            write(image, "<f4", "3, 128, 128", false, pixels);
            ByteBuffer labels = ByteBuffer.allocate(128 * 128 * 2).order(ByteOrder.BIG_ENDIAN);
            for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++)
                labels.putShort((short) (x >= 40 && x < 60 && y >= 40 && y < 60 ? 300 : 0));
            write(mask, ">u2", "128, 128", false, labels.array());
            try (NpyImageFile numpy = new NpyImageFile(mask)) {
                numpy.resolve(128, 128);
                assertEquals(300, numpy.value(42, 43, 0), 0);
                assertEquals(0, numpy.value(0, 0, 0), 0);
            }
        }
        File source = new File(root, "train/images/sample.npy");
        byte[] before = Files.readAllBytes(source.toPath());
        File models = temporary.newFolder("software", "models");
        YoloDatasetPreparationResult prepared = YoloDatasetPreparer.prepareDetailed(root.getAbsolutePath(),
                "test", models.getAbsolutePath(), 640, message -> {});
        File generated = prepared.getYaml().getParentFile();
        Map<?, ?> references = new Gson().fromJson(new String(Files.readAllBytes(new File(generated, "jdll_samples.json").toPath()), StandardCharsets.UTF_8), Map.class);
        assertEquals(2, references.size());
        for (Object path : references.keySet()) assertFalse(new File(path.toString()).exists());
        assertEquals(1, new File(generated, "labels/train").list().length);
        assertEquals(1, new File(generated, "labels/val").list().length);
        assertArrayEquals(before, Files.readAllBytes(source.toPath()));
        YoloDatasetPreparationResult reused = YoloDatasetPreparer.prepareDetailed(prepared.getYaml().getAbsolutePath(),
                "again", models.getAbsolutePath(), 640, message -> {});
        assertEquals(prepared.getYaml(), reused.getYaml());
        assertTrue(YoloImageFiles.canReadImage(source));
        String python = System.getenv("JDLL_PYTHON");
        if (python != null && !python.isEmpty()) {
            File output = new File(temporary.getRoot(), "model/model.pt");
            Files.createDirectories(output.getParentFile().toPath());
            YoloTrainingConfig config = new YoloTrainingConfig("model", prepared.getYaml().getAbsolutePath(), 1, 128,
                    false, null, "yolo26n.yaml", models.getAbsolutePath(), output.getAbsolutePath(), 1, "cpu");
            config.writeConfig(prepared.getYaml());
            java.lang.reflect.Method build = io.bioimage.modelrunner.model.special.yolo.Yolo.class.getDeclaredMethod(
                    "buildTrainingCode", int.class, String.class, String.class, String.class, int.class, int.class,
                    String.class, String.class, Map.class, Object.class);
            build.setAccessible(true);
            Map<String, Object> options = new java.util.LinkedHashMap<>();
            options.put("amp", false);
            options.put("pretrained", false);
            options.put("close_mosaic", 0);
            options.put("jdll_preview_assets", new File(output.getParentFile(), "validation_images").getAbsolutePath());
            String script = (String) build.invoke(null, 1, "yolo26n.yaml", prepared.getYaml().getAbsolutePath(),
                    output.getAbsolutePath(), 128, 1, "", "cpu", options, Integer.valueOf(1));
            File task = new File(temporary.getRoot(), "task.py");
            Files.write(task.toPath(), script.getBytes(StandardCharsets.UTF_8));
            File log = new File(temporary.getRoot(), "python.log");
            Process process = new ProcessBuilder(python, "src/test/python/test_yolo_bridge.py", task.getAbsolutePath(), output.getAbsolutePath())
                    .redirectErrorStream(true).redirectOutput(log).start();
            assertTrue("YOLO bridge timed out", process.waitFor(120, java.util.concurrent.TimeUnit.SECONDS));
            assertEquals(new String(Files.readAllBytes(log.toPath()), StandardCharsets.UTF_8), 0, process.exitValue());
            assertTrue(output.isFile());
            try (InferenceCode model = new InferenceCode(output.getAbsolutePath())) {
                Files.write(task.toPath(), model.code().getBytes(StandardCharsets.UTF_8));
                process = new ProcessBuilder(python, "src/test/python/test_yolo_bridge.py", task.getAbsolutePath(),
                        output.getAbsolutePath(), "inference").redirectErrorStream(true).redirectOutput(log).start();
                assertTrue("YOLO inference timed out", process.waitFor(120, java.util.concurrent.TimeUnit.SECONDS));
                assertEquals(new String(Files.readAllBytes(log.toPath()), StandardCharsets.UTF_8), 0, process.exitValue());
            }
        }
        assertArrayEquals(before, Files.readAllBytes(source.toPath()));
    }

    @Test public void handlesFortranOrderAndStopsAtFirstInvalidMask() throws Exception {
        File image = temporary.newFile("image.npy");
        ByteBuffer pixels = ByteBuffer.allocate(128 * 160 * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int y = 0; y < 160; y++) for (int x = 0; x < 128; x++) pixels.putFloat(x + y * 1000);
        write(image, "<f4", "128, 160", true, pixels.array());
        try (NpyImageFile numpy = new NpyImageFile(image)) {
            numpy.resolve(128, 160);
            assertEquals(7003, numpy.value(3, 7, 0), 0);
        }
        File root = temporary.newFolder("bad-source");
        File source = new File(root, "images/a.npy");
        write(source, "<f4", "128, 160", false, new byte[128 * 160 * 4]);
        File mask = new File(root, "labels/a_masks.npy");
        ByteBuffer bad = ByteBuffer.allocate(128 * 160 * 4).order(ByteOrder.LITTLE_ENDIAN);
        bad.putFloat(0.5f);
        write(mask, "<f4", "128, 160", false, bad.array());
        try {
            YoloDatasetPreparer.prepareDetailed(root.getAbsolutePath(), "bad", temporary.getRoot().getAbsolutePath(), 640, message -> {});
            fail("Non-integer masks must not be silently cast");
        } catch (IOException e) {
            assertTrue(e.getMessage().contains(mask.getAbsolutePath()));
            assertTrue(e.getMessage().contains(NpyImageFile.DOCUMENTATION_URL));
        }
    }

    private static final class InferenceCode extends io.bioimage.modelrunner.model.special.yolo.Yolo implements AutoCloseable {
        InferenceCode(String weights) throws org.apposed.appose.BuildException { super(weights, "cpu"); }

        String code() throws IOException {
            float[] values = new float[128 * 128];
            for (int i = 0; i < values.length; i++) values[i] = i % 256;
            io.bioimage.modelrunner.tensor.Tensor<net.imglib2.type.numeric.real.FloatType> tensor =
                    io.bioimage.modelrunner.tensor.Tensor.build("image", "xy", net.imglib2.img.array.ArrayImgs.floats(values, 128, 128));
            return buildModelCode() + "\nimport numpy as np\ncreated_shms = []\n"
                    + createInputsCode(Arrays.asList(tensor), Arrays.asList("image"));
        }

        @Override public void close() {
            for (io.bioimage.modelrunner.tensor.shm.SharedMemoryArray array : inShmaList) array.close();
            super.close();
        }
    }

    private static void write(File file, String dtype, String shape, boolean fortran, byte[] values) throws Exception {
        Files.createDirectories(file.getParentFile().toPath());
        String text = "{'descr': '" + dtype + "', 'fortran_order': " + (fortran ? "True" : "False")
                + ", 'shape': (" + shape + "), }";
        int padding = (64 - (10 + text.length() + 1) % 64) % 64;
        char[] spaces = new char[padding];
        Arrays.fill(spaces, ' ');
        byte[] header = (text + new String(spaces) + "\n").getBytes(StandardCharsets.US_ASCII);
        ByteBuffer out = ByteBuffer.allocate(10 + header.length + values.length).order(ByteOrder.LITTLE_ENDIAN);
        out.put(new byte[] {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0}).putShort((short) header.length).put(header).put(values);
        Files.write(file.toPath(), out.array());
    }
}
