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

public enum DenoisingNoiseStructure {
    AUTO("auto", "Auto"),
    LOCAL("local_isotropic", "Local / isotropic"),
    HORIZONTAL("horizontal", "Horizontal"),
    VERTICAL("vertical", "Vertical"),
    PERIODIC("periodic_grid", "Periodic / grid");
    private final String id;
    private final String label;
    DenoisingNoiseStructure(String id, String label) { this.id = id; this.label = label; }
    public String getId() { return id; }
    @Override public String toString() { return label; }
}
