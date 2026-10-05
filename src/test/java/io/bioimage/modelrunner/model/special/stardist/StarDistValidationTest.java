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

import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import io.bioimage.modelrunner.model.special.common.TrainingCodeUtils;

public class StarDistValidationTest {
    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();
    @Test
    public void generatedTrainingUsesSharedValidationWithoutAutomaticFinalFullPass() {
        Map<String, Object> config = new LinkedHashMap<String, Object>(StarDist.defaultTrainingConfig(2));
        config.put("n_dim", 3);
        String code = StarDist.buildTrainingCode("/tmp/data", null, "/tmp/model", "cpu",
                "grayscale", "grayscale", 0.15, config);
        assertTrue(code.contains("validation_options = config.pop('validation', {})"));
        assertTrue(code.contains("stardist_validation.train_with_validation"));
        assertTrue(code.contains("'val/object_f1': logs.get('val_object_f1')"));
        assertFalse(code.contains("full_volume = n_dim == 3 and current_epoch == state['total_epochs']"));
        assertFalse(code.contains("self.model_ref.predict_instances"));
        assertTrue(code.contains("best_path.exists() and n_dim == 2"));
    }

    @Test
    public void objectF1IsSeparateFromLearningRateAndLoss() {
        StardistTrainingProgress progress = new StardistTrainingProgress(1, 10, 2, 20,
                Collections.singletonMap(StardistTrainingProgress.VAL_TOTAL_LOSS, 0.3),
                Collections.singletonMap(StardistTrainingProgress.OBJECT_F1, 0.8));
        assertEquals(0.8, progress.getObjectF1(), 0);
        assertEquals(0.3, progress.getValidationTotalLoss(), 0);
        assertNull(progress.getLearningRate());
    }

    /** Uses the installed StarDist stack, never downloads an environment or trains a real dataset. */
    @Test
    public void smallCpuBackendRegressionSuite() throws Exception {
        String python = System.getenv("JDLL_STARDIST_TEST_PYTHON");
        assumeTrue("Set JDLL_STARDIST_TEST_PYTHON to run backend integration tests", python != null);
        List<Map<String, Object>> tasks = new ArrayList<Map<String, Object>>();
        for (int example = 0; example < 4; example++) {
            int dimensions = 2 + example % 2;
            File data = temporaryFolder.newFolder("data" + example);
            File output = temporaryFolder.newFolder("model" + example);
            File script = temporaryFolder.newFile("train" + example + ".py");
            Map<String, Object> config = StarDist.defaultTrainingConfig(1);
            config.put("n_dim", dimensions);
            config.put("n_rays", 8);
            config.put("unet_n_depth", 1);
            config.put("unet_n_filter_base", 2);
            config.put("net_conv_after_unet", 2);
            config.put("train_batch_size", 1);
            config.put("train_steps_per_epoch", 1);
            config.put("train_patch_size", dimensions == 3 ? Arrays.asList(4, 16, 16) : Arrays.asList(16, 16));
            config.put("unet_pool", dimensions == 3 ? Arrays.asList(1, 2, 2) : Arrays.asList(2, 2));
            config.put("grid", dimensions == 3 ? Arrays.asList(1, 1, 1) : Arrays.asList(1, 1));
            config.put("validation_preview_count", 1);
            Map<String, Object> validation = new LinkedHashMap<String, Object>();
            validation.put("minimum_batches", 1);
            validation.put("minimum_samples", 2);
            config.put("validation", validation);
            Files.write(script.toPath(), StarDist.buildTrainingCode(data.getAbsolutePath(), null,
                    output.getAbsolutePath(), "cpu", "grayscale", "grayscale", 0.15, config)
                    .getBytes(StandardCharsets.UTF_8));
            Map<String, Object> task = new LinkedHashMap<String, Object>();
            task.put("dimensions", dimensions);
            task.put("single_source", example >= 2);
            task.put("script", script.getAbsolutePath());
            task.put("dataset", data.getAbsolutePath());
            task.put("output", output.getAbsolutePath());
            tasks.add(task);
        }
        ProcessBuilder builder = new ProcessBuilder(python, "src/test/python/test_stardist_validation.py").inheritIO();
        builder.environment().put("JDLL_GENERATED_TRAINING_TASKS", TrainingCodeUtils.toJson(tasks));
        Process process = builder.start();
        boolean completed = process.waitFor(120, TimeUnit.SECONDS);
        if (!completed) process.destroyForcibly();
        assertTrue("StarDist validation tests timed out", completed);
        assertEquals(0, process.exitValue());
    }
}
