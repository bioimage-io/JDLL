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
 * #L%
 */
package io.bioimage.modelrunner.model.special.denoise;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/** Structured progress emitted by the Python denoising backend. */
public final class DenoisingProgress {

    private final String phase;
    private final String message;
    private final long current;
    private final long maximum;
    private final Map<String, Object> info;

    public DenoisingProgress(String phase, String message, long current, long maximum,
            Map<String, Object> info) {
        this.phase = phase == null ? "" : phase;
        this.message = message == null ? "" : message;
        this.current = current;
        this.maximum = maximum;
        this.info = info == null ? Collections.emptyMap()
                : Collections.unmodifiableMap(new LinkedHashMap<String, Object>(info));
    }

    public String getPhase() { return phase; }
    public String getMessage() { return message; }
    public long getCurrent() { return current; }
    public long getMaximum() { return maximum; }
    public Map<String, Object> getInfo() { return info; }
}
