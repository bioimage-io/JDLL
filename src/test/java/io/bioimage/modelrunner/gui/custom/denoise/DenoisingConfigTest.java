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
}
