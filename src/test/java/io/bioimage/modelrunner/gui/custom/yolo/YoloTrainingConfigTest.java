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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.Test;

public class YoloTrainingConfigTest {

    @Test
    public void writesPortableFormatZeroConfig() throws Exception {
        File root = Files.createTempDirectory("jdll-yolo-config").toFile();
        File output = new File(root, "model/model.pt");
        YoloTrainingConfig config = new YoloTrainingConfig("model", root.getAbsolutePath(), 12, 640,
                false, null, "yolo26m.yaml", root.getAbsolutePath(), output.getAbsolutePath(), 1, "cuda");

        Map<String, Object> json = config.toConfigMap(new File(root, "data.yaml"));
        assertEquals("yolo", json.get("framework"));
        assertEquals(0, ((Number) json.get("format_version")).intValue());
        assertFalse(json.containsKey("validation_preview"));
        assertFalse(json.containsKey("outputs"));
        assertFalse(json.containsKey("logging"));

        Map<String, Object> training = map(json.get("training"));
        assertEquals(12, ((Number) training.get("epochs")).intValue());
        assertEquals(640, ((Number) training.get("imgsz")).intValue());
        assertEquals(0, ((Number) training.get("seed")).intValue());
        assertEquals(Boolean.TRUE, training.get("deterministic"));
        assertEquals(64, ((Number) training.get("nbs")).intValue());

        Map<String, Object> runtime = map(json.get("runtime"));
        assertEquals("cuda", runtime.get("requested_device"));
        assertEquals("auto", runtime.get("requested_batch"));
        assertNull(runtime.get("resolved_batch"));
        assertEquals(1, ((Number) runtime.get("minimum_batch")).intValue());

        Map<String, Object> software = map(json.get("software"));
        assertTrue(software.containsKey("jdll"));
        assertTrue(software.containsKey("deepicy"));
        assertTrue(software.containsKey("python"));
        deleteTree(root);
    }

    @Test
    public void appliesCustomTrainingOptionsAndRelativeArchitecture() throws Exception {
        File root = Files.createTempDirectory("jdll-yolo-custom").toFile();
        File yaml = new File(root, "architecture.yaml");
        Files.write(yaml.toPath(), "nc: 1\n".getBytes(StandardCharsets.UTF_8));
        File custom = new File(root, "config.json");
        Files.write(custom.toPath(), (
                "{\"framework\":\"yolo\",\"format_version\":0,"
                + "\"model\":{\"source\":\"architecture.yaml\"},"
                + "\"training\":{\"imgsz\":512,\"lr0\":0.002,\"augmentations\":{\"flipud\":0.5}},"
                + "\"runtime\":{\"requested_batch\":3}}")
                .getBytes(StandardCharsets.UTF_8));

        YoloTrainingConfig config = new YoloTrainingConfig("custom", root.getAbsolutePath(), 20, 640,
                false, null, custom.getAbsolutePath(), root.getAbsolutePath(),
                new File(root, "custom/custom.pt").getAbsolutePath(), 1, "cpu");

        assertEquals(yaml.getAbsolutePath(), config.getTrainingModelSource());
        assertEquals(512, config.getImageSize());
        assertEquals(3, ((Number) config.getRequestedBatch()).intValue());
        assertEquals(0.002, ((Number) config.getBackendTrainingOptions().get("lr0")).doubleValue(), 0.0);
        assertEquals(0.5, ((Number) config.getBackendTrainingOptions().get("flipud")).doubleValue(), 0.0);
        assertEquals(20, config.getEpochs());
        deleteTree(root);
    }

    @Test
    public void exposesAllFiveScratchArchitectures() {
        LinkedHashMap<String, String> entries = YoloModelRegistry.buildScratchArchitectureEntries();
        assertEquals(5, entries.size());
        assertTrue(entries.containsValue("yolo26n.yaml"));
        assertTrue(entries.containsValue("yolo26s.yaml"));
        assertTrue(entries.containsValue("yolo26m.yaml"));
        assertTrue(entries.containsValue("yolo26l.yaml"));
        assertTrue(entries.containsValue("yolo26x.yaml"));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object value) {
        return (Map<String, Object>) value;
    }

    private static void deleteTree(File file) {
        if (file == null || !file.exists()) return;
        File[] children = file.listFiles();
        if (children != null) {
            for (File child : children) deleteTree(child);
        }
        file.delete();
    }
}
