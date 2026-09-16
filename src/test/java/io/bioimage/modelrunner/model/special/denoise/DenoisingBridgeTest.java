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
package io.bioimage.modelrunner.model.special.denoise;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class DenoisingBridgeTest {
    @Test
    public void forwardsEveryEffortToBothPublicEntryPointsWithCallbacks() {
        for (String effort : new String[] {"quick", "balanced", "balanced_high", "thorough"}) {
            DenoisingConfig config = DenoisingConfig.builder().method("zs_n2n").effort(effort).build();
            for (boolean preview : new boolean[] {false, true}) {
                String code = Denoising.denoiseCallCode("input_array", config, preview);
                String callable = preview ? "jdll_denoise_preview" : "jdll_denoise_run";
                assertTrue(code.contains(callable
                        + "(input_array, _jdll_denoise_config, callbacks=_jdll_denoise_callback)"));
                assertTrue(code.contains("\"method\":\"zs_n2n\""));
                assertTrue(code.contains("\"effort\":\"" + effort + "\""));
                assertTrue(code.contains("\"method_config\":{}"));
                assertTrue(code.contains("\"low\":0.0,\"high\":100.0"));
                assertTrue(code.contains("info=event"));
                assertFalse(code.contains("hidden_width"));
                assertFalse(code.contains("batch_pairs"));
            }
        }
    }

    @Test
    public void guardsAgainstOldPackagesAndMissingEfforts() {
        for (String effort : new String[] {"quick", "balanced", "balanced_high", "thorough"}) {
            String code = Denoising.backendCompatibilityCode(DenoisingConfig.builder()
                    .method("zs_n2n").effort(effort).build());
            assertTrue(code.contains("effort_presets_version"));
            assertTrue(code.contains("_jdll_zs_version < 1"));
            assertTrue(code.contains("'" + effort + "' not in _jdll_zs_capability.get('efforts', [])"));
            assertTrue(code.contains("Update jdll-denoise"));
        }
        for (String method : new String[] {"noise2fast", "structn2v", "bm3d_bm4d"}) {
            assertEquals("", Denoising.backendCompatibilityCode(
                    DenoisingConfig.builder().method(method).build()));
        }
    }
}
