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
package io.bioimage.modelrunner.gui.custom.unet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Map;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class UnetTrainingServiceTest {

    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void customScratchConfigProvidesArchitectureAndLearningRate() throws Exception {
        File configFile = temporaryFolder.newFile("config.json");
        Files.write(configFile.toPath(), (
                "{\"format\":\"jdll-unet\",\"framework\":\"unet\","
                + "\"architecture\":\"resenc-medium-2d\","
                + "\"training\":{\"learning_rate\":0.002}}")
                .getBytes(StandardCharsets.UTF_8));
        UnetTrainingConfig config = config(false, null, configFile.getAbsolutePath());

        Map<String, Object> request = UnetTrainingService.toPythonConfig(config, temporaryFolder.getRoot());

        assertEquals("scratch", request.get("starting_point"));
        assertEquals("resenc-medium-2d", request.get("architecture"));
        assertEquals(0.002d, ((Number) request.get("learning_rate")).doubleValue(), 0.0d);
        assertFalse(request.containsKey("base_model"));
    }

    @Test
    public void fineTuningLeavesArchitectureAndLearningRateResolutionToBackend() throws Exception {
        File baseModel = temporaryFolder.newFile("model.pt");
        UnetTrainingConfig config = config(true, baseModel.getAbsolutePath(), null);

        Map<String, Object> request = UnetTrainingService.toPythonConfig(config, temporaryFolder.getRoot());

        assertEquals("fine_tune", request.get("starting_point"));
        assertEquals(baseModel.getAbsolutePath(), request.get("base_model"));
        assertEquals("auto", request.get("learning_rate"));
        assertFalse(request.containsKey("architecture"));
    }

    private UnetTrainingConfig config(boolean fineTune, String baseModel, String scratchArchitecture) {
        File output = new File(temporaryFolder.getRoot(), "output");
        return new UnetTrainingConfig("model", temporaryFolder.getRoot().getAbsolutePath(), 5,
                fineTune, baseModel, scratchArchitecture, temporaryFolder.getRoot().getAbsolutePath(),
                output.getAbsolutePath(), "cpu");
    }
}
