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

/** User-facing denoising methods and stable backend identifiers. */
public enum DenoisingMethod {
    CONSERVATIVE("bm3d_bm4d", "Conservative - BM3D/BM4D", false),
    ADAPTIVE("zs_n2n", "Adaptive - ZS-N2N", true),
    FAST("noise2fast", "Fast - Noise2Fast", true),
    CORRELATED("structn2v", "Correlated - StructN2V", true);

    private final String id;
    private final String label;
    private final boolean accelerationSupported;

    DenoisingMethod(String id, String label, boolean accelerationSupported) {
        this.id = id;
        this.label = label;
        this.accelerationSupported = accelerationSupported;
    }

    public String getId() { return id; }
    public boolean supportsAcceleration() { return accelerationSupported; }
    @Override public String toString() { return label; }
}
