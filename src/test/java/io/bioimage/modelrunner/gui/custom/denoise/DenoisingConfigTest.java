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
package io.bioimage.modelrunner.gui.custom.denoise;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Map;

import org.junit.Test;

import io.bioimage.modelrunner.model.special.denoise.DenoisingConfig;

public class DenoisingConfigTest {
    @Test
    public void usesProductionDefaultsAndNormalizesDevice() {
        DenoisingConfig config = DenoisingConfig.builder().device("unsupported").build();
        assertEquals("noise2fast", config.getMethod());
        assertEquals("balanced", config.getEffort());
        assertEquals("cpu", config.getDevice());
        assertEquals("auto", config.getNoiseStructure());
        assertEquals(5489L, config.getSeed());
        assertFalse((Boolean) config.toMap().get("allow_cpu_fallback"));
        assertFalse(config.toMap().containsKey("noise_structure"));
        assertTrue(((Map<?, ?>) config.toMap().get("method_config")).isEmpty());
        assertEquals("auto", ((Map<?, ?>) config.toMap().get("tiling")).get("enabled"));
    }

    @Test
    public void nestsStructN2VConfigurationUnderMethodConfig() {
        DenoisingConfig config = DenoisingConfig.builder()
                .method("structn2v").noiseStructure("horizontal").build();

        assertEquals("horizontal",
                ((Map<?, ?>) config.toMap().get("method_config")).get("noise_structure"));
    }

    @Test
    public void forwardsAllZsN2NEffortsWithoutDuplicatingPythonPresets() {
        for (DenoisingEffort effort : DenoisingEffort.values()) {
            for (String device : new String[] {"cpu", "cuda", "mps"}) {
                DenoisingConfig config = DenoisingConfig.builder()
                        .method("zs_n2n").effort(effort.getId()).device(device).axes("xyc").build();
                Map<String, Object> request = config.toMap();
                assertEquals("zs_n2n", request.get("method"));
                assertEquals(effort.getId(), request.get("effort"));
                assertEquals(device, request.get("device"));
                assertEquals("xyc", request.get("axes"));
                assertTrue(((Map<?, ?>) request.get("method_config")).isEmpty());
                Map<?, ?> normalization = (Map<?, ?>) request.get("normalization");
                assertEquals(0d, normalization.get("low"));
                assertEquals(100d, normalization.get("high"));
            }
        }
    }

    @Test
    public void normalizesBalancedHighId() {
        assertEquals("balanced_high", DenoisingConfig.builder()
                .method("ZS_N2N").effort(" Balanced-High ").build().getEffort());
    }

    @Test
    public void preservesOtherMethodsNormalization() {
        for (String method : new String[] {"noise2fast", "structn2v", "bm3d_bm4d"}) {
            Map<?, ?> normalization = (Map<?, ?>) DenoisingConfig.builder()
                    .method(method).build().toMap().get("normalization");
            assertEquals(0.1d, normalization.get("low"));
            assertEquals(99.9d, normalization.get("high"));
        }
    }

    @Test
    public void rejectsBalancedHighForOtherMethods() {
        for (String method : new String[] {"noise2fast", "structn2v", "bm3d_bm4d"}) {
            org.junit.Assert.assertThrows(IllegalArgumentException.class,
                    () -> DenoisingConfig.builder().method(method).effort("balanced_high").build());
        }
    }
}
