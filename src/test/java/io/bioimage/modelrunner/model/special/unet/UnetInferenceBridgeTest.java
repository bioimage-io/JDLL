package io.bioimage.modelrunner.model.special.unet;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeNotNull;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.concurrent.TimeUnit;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class UnetInferenceBridgeTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test public void outputSelectionFollowsTheTaskRatherThanAvailableAuxiliaryOutputs() {
        String code = Unet.inferenceCode("model.pt", null, "input_image");
        assertFalse(code.contains("connected_components"));
        assertTrue(code.contains("if _jdll_unet_task == 'instance_friendly':"));
        assertTrue(code.contains("else:" + System.lineSeparator()
                + "  _jdll_unet_labels = _jdll_unet_outputs['mask']"));
        assertFalse(code.contains("if 'labels' in"));
        assertFalse(code.contains("['object_size']"));
        assertTrue(Unet.inferenceCode("model.pt", 12.0, "input_image").contains("['object_size'] = 12.0"));
    }

    @Test public void generatedInferencePreservesBinaryClassAndInstanceValues() throws Exception {
        String python = System.getenv("JDLL_UNET_TEST_PYTHON");
        assumeNotNull(python);
        File script = temporary.newFile("inference.py");
        Files.write(script.toPath(), Unet.inferenceCode("model.pt", null, "input_image")
                .getBytes(StandardCharsets.UTF_8));
        Process process = new ProcessBuilder(python, "-B", "src/test/python/test_unet_inference_bridge.py",
                script.toString()).inheritIO().start();
        boolean finished = process.waitFor(30, TimeUnit.SECONDS);
        if (!finished) process.destroyForcibly();
        assertTrue("UNet inference output tests timed out", finished);
        assertEquals(0, process.exitValue());
    }
}
