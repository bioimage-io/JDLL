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
package io.bioimage.modelrunner.model.special.stardist;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.junit.Test;

public class StarDistImageLoadingTest {
    @Test
    public void usesSeparateReadersForTiffAndRasterFormats() {
        for (int dimensions : new int[] {2, 3}) {
            String code = trainingCode(dimensions);
            assertTrue(code.contains("from PIL import Image"));
            assertTrue(code.contains("from tifffile import TiffFile"));
            assertFalse(code.contains("from tifffile import TiffFile, imread"));
            assertFalse(code.contains("from tifffile import imread"));
            assertTrue(code.contains("Image.open(path, formats=[file_format])"));
            assertTrue(code.contains("raw_x, x_axes = _read_array(img_path)"));
            assertTrue(code.contains("raw_y, y_axes = _read_array(mask_path, is_mask=True)"));
        }
    }

    @Test
    public void preservesTiffSeriesAxesAndAllAcceptedImageExtensions() {
        String code = trainingCode(3);
        assertTrue(code.contains("if file_format == 'TIFF':"));
        assertTrue(code.contains("series = tif.series[0]"));
        assertTrue(code.contains("return array, str(series.axes).upper()"));
        assertTrue(code.contains("IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}"));
    }

    /** Set JDLL_TEST_PYTHON to an environment with numpy, Pillow and tifffile. */
    @Test
    public void decodesRealImagesWithTheGeneratedPythonLoader() throws Exception {
        String python = System.getenv("JDLL_TEST_PYTHON");
        assumeTrue("Set JDLL_TEST_PYTHON to run image decoding tests", python != null && !python.isEmpty());
        Path script = Files.createTempFile("stardist-training-", ".py");
        try {
            Files.write(script, trainingCode(3).getBytes(StandardCharsets.UTF_8));
            Process process = new ProcessBuilder(python, "src/test/python/test_stardist_image_loading.py",
                    script.toString()).inheritIO().start();
            boolean finished = process.waitFor(90, TimeUnit.SECONDS);
            if (!finished) process.destroyForcibly();
            assertTrue("Python image decoding tests timed out", finished);
            assertEquals("Python image decoding tests failed", 0, process.exitValue());
        } finally {
            Files.deleteIfExists(script);
        }
    }

    private static String trainingCode(int dimensions) {
        Map<String, Object> config = new LinkedHashMap<String, Object>(StarDist.defaultTrainingConfig(2));
        config.put("n_dim", dimensions);
        return StarDist.buildTrainingCode("/tmp/dataset", null, "/tmp/output",
                "cpu", "rgb", "grayscale", 0.15d, config, "");
    }
}
