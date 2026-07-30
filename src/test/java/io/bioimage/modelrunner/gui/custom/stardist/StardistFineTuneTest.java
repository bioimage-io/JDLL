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
package io.bioimage.modelrunner.gui.custom.stardist;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.gui.custom.stardist.StardistModelRegistry.FineTuneSource;
import io.bioimage.modelrunner.model.special.stardist.StarDist;

public class StardistFineTuneTest {

    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void resolvesOnlyCompleteTwoDimensionalModels() throws Exception {
        File model = temporaryFolder.newFolder("model");
        File weights = new File(model, "weights_best.h5");
        Files.write(weights.toPath(), new byte[] {1});
        Files.write(new File(model, "config.json").toPath(),
                "{\"n_dim\":2,\"axes\":\"YXC\",\"n_channel_in\":1,\"n_rays\":32,\"grid\":[1,1],\"backbone\":\"unet\"}"
                        .getBytes(StandardCharsets.UTF_8));

        FineTuneSource fromDirectory = StardistModelRegistry.resolveFineTuneSource(model.getAbsolutePath());
        FineTuneSource fromWeights = StardistModelRegistry.resolveFineTuneSource(weights.getAbsolutePath());

        assertNotNull(fromDirectory);
        assertNotNull(fromWeights);
        assertTrue(StardistModelRegistry.isSelectableFineTuneSource(weights.getAbsolutePath()));

        File invalid = temporaryFolder.newFolder("invalid");
        Files.write(new File(invalid, "weights_best.h5").toPath(), new byte[] {1});
        Files.write(new File(invalid, "config.json").toPath(),
                "{\"n_dim\":3,\"axes\":\"ZYXC\",\"n_channel_in\":1}".getBytes(StandardCharsets.UTF_8));
        assertNull(StardistModelRegistry.resolveFineTuneSource(invalid.getAbsolutePath()));
    }

    @Test
    public void generatedTaskAdaptsChannelsAndPreservesLastCheckpoint() {
        Map<String, Object> config = new LinkedHashMap<String, Object>(StarDist.defaultTrainingConfig(2));
        config.put("_jdll_fine_tune_weights", "/tmp/weights_initial.h5");
        config.put("_jdll_fine_tune_source_config",
                new LinkedHashMap<String, Object>(StarDist.defaultTrainingConfig(2)));

        String code = StarDist.buildTrainingCode("/tmp/data", null, "/tmp/output",
                "cpu", "rgb", "grayscale", 0.15d, config, "/tmp/cancel");

        assertTrue(code.contains("def _adapt_input_kernel"));
        assertTrue(code.contains("Adapted the first convolution"));
        assertTrue(code.contains("Fine-tuning baseline validation"));
        assertTrue(code.contains("model.optimize_thresholds"));
        assertFalse(code.contains("model.keras_model.save_weights(str(last_path))"));
    }

    @Test
    public void resolvesThreeDimensionalModelsForThreeDimensionalDatasetsOnly() throws Exception {
        File model = temporaryFolder.newFolder("model-3d");
        File weights = new File(model, "weights_best.h5");
        Files.write(weights.toPath(), new byte[] {1});
        Files.write(new File(model, "config.json").toPath(),
                ("{\"n_dim\":3,\"axes\":\"ZYXC\",\"n_channel_in\":1,\"n_rays\":96,"
                        + "\"grid\":[1,2,2],\"backbone\":\"unet\",\"anisotropy\":[4.0,1.0,1.0]}")
                        .getBytes(StandardCharsets.UTF_8));

        assertNotNull(StardistModelRegistry.resolveFineTuneSource(model.getAbsolutePath()));
        assertTrue(StardistModelRegistry.isSelectableFineTuneSource(model.getAbsolutePath(), 3));
        assertFalse(StardistModelRegistry.isSelectableFineTuneSource(model.getAbsolutePath(), 2));
    }

    @Test
    public void generatedThreeDimensionalTaskPreservesVolumesAndResolvesAnisotropy() {
        Map<String, Object> config = new LinkedHashMap<String, Object>(StarDist.defaultTrainingConfig(2));
        config.put("n_dim", 3);
        config.put("axes", "ZYXC");
        config.put("n_rays", 96);
        config.put("grid", java.util.Arrays.asList(1, 2, 2));
        config.put("train_patch_size", java.util.Arrays.asList(16, 64, 64));
        config.put("_jdll_auto_anisotropy", true);

        String code = StarDist.buildTrainingCode("/tmp/data", null, "/tmp/output",
                "cpu", "grayscale", "grayscale", 0.15d, config, "/tmp/cancel");

        assertTrue(code.contains("from stardist.models import Config2D, Config3D, StarDist2D, StarDist3D"));
        assertTrue(code.contains("def _resolve_anisotropy"));
        assertTrue(code.contains("OME physical voxel spacing"));
        assertTrue(code.contains("training-mask object extents"));
        assertTrue(code.contains("def _save_object_statistics"));
        assertTrue(code.contains("dataset_statistics.json"));
        assertTrue(code.contains("model_metadata.json"));
        assertTrue(code.contains("equivalent_sphere_diameter"));
        assertTrue(code.contains("StarDistData3D"));
        assertTrue(code.contains("sample['initial_plane']"));
        assertTrue(code.contains("n_tiles=preview_tiles"));
        assertFalse(code.contains("y = y[..., 0]"));
    }
}
