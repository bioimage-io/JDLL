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
package io.bioimage.modelrunner.model.special.unet;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import org.apposed.appose.TaskEvent;
import org.apposed.appose.Service.ResponseType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import io.bioimage.modelrunner.model.special.common.TrainingCodeUtils;
import io.bioimage.modelrunner.model.special.stardist.StarDist;

public class UnetValidationBridgeTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    private String code(Map<String, Object> config) throws Exception {
        Method method = Unet.class.getDeclaredMethod("buildTrainingCode", Map.class);
        method.setAccessible(true);
        return (String) method.invoke(null, config);
    }

    @Test
    public void forwardsControlSeparatelyFromTrainingConfiguration() throws Exception {
        String script = code(Collections.singletonMap("_jdll_validation_requests", "/tmp/requests"));
        assertTrue(script.contains(".pop('_jdll_validation_requests', None)"));
        assertTrue(script.contains("task=task, control=_jdll_control"));
        assertFalse(script.contains("resume_from"));
    }

    @Test
    public void structuredEventsPreserveExistingPreviewConsumers() throws Exception {
        for (Class<?> bridge : new Class<?>[] {Unet.class, StarDist.class}) {
            Method handler = bridge.getDeclaredMethod("handleTrainingEvent", TaskEvent.class,
                    Consumer.class, Consumer.class, Consumer.class, Consumer.class);
            handler.setAccessible(true);
            AtomicInteger progress = new AtomicInteger(), previews = new AtomicInteger();
            List<Map<String, Object>> events = new ArrayList<>();
            for (String type : new String[] {"validation_plan", "validation", "checkpoint", "preview", "full_validation"}) {
                Map<String, Object> info = new LinkedHashMap<>();
                info.put("type", type);
                info.put("epoch", 1);
                info.put("preview_path", "/tmp/preview.json");
                TaskEvent event = new TaskEvent(null, ResponseType.UPDATE, null, 5, 100, info);
                handler.invoke(null, event, (Consumer<Object>) p -> progress.incrementAndGet(),
                        (Consumer<Object>) p -> previews.incrementAndGet(), null,
                        (Consumer<Map<String, Object>>) events::add);
            }
            assertEquals(5, events.size());
            assertEquals(1, previews.get());
            assertEquals("Validation must not advance training iterations", 0, progress.get());
        }
    }

    @Test
    public void generatedTaskRunsAgainstCurrentPythonContract() throws Exception {
        String python = System.getenv("JDLL_UNET_TEST_PYTHON");
        assumeTrue("Set JDLL_UNET_TEST_PYTHON to test the installed jdll-unet API", python != null);
        List<Map<String, Object>> fixtures = new ArrayList<>();
        for (String dimensions : new String[] {"2.5d", "3d"}) {
            Map<String, Object> config = new LinkedHashMap<>();
            config.put("model_name", "test");
            config.put("output_dir", temporary.newFolder("model-" + dimensions).toString());
            config.put("dataset_path", temporary.newFolder("data-" + dimensions).toString());
            config.put("_jdll_validation_requests", temporary.newFolder("requests-" + dimensions).toString());
            config.put("architecture", "resenc-tiny-" + dimensions);
            config.put("patch_size", "3d".equals(dimensions) ? Arrays.asList(8, 16, 16) : Arrays.asList(16, 16));
            config.put("device", "cpu");
            config.put("batch_size", 1);
            config.put("effective_batch_size", 1);
            config.put("steps_per_epoch", 1);
            config.put("epochs", 2);
            config.put("task", "binary_semantic");
            config.put("preview_count", 1);
            config.put("context_slices", 3);
            config.put("instance_scale_normalization", Collections.singletonMap("enabled", false));
            Map<String, Object> validation = new LinkedHashMap<>();
            validation.put("minimum_batches", 2);
            validation.put("minimum_samples", 2);
            config.put("validation", validation);
            java.io.File script = temporary.newFile("training-" + dimensions + ".py");
            Files.write(script.toPath(), code(config).getBytes(StandardCharsets.UTF_8));
            config.put("script", script.toString());
            fixtures.add(config);
        }
        ProcessBuilder builder = new ProcessBuilder(python, "src/test/python/test_unet_validation_bridge.py").inheritIO();
        builder.environment().put("JDLL_UNET_TASKS", TrainingCodeUtils.toJson(fixtures));
        Process process = builder.start();
        boolean finished = process.waitFor(180, TimeUnit.SECONDS);
        if (!finished) process.destroyForcibly();
        assertTrue("UNet validation bridge tests timed out", finished);
        assertEquals(0, process.exitValue());
    }
}
