/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * #L%
 */
package io.bioimage.modelrunner.gui.custom;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.function.BooleanSupplier;

/** Host-owned, background-only reader for series zero. Coordinates are native image pixels. */
public interface PreviewReader extends AutoCloseable {
    final class Info {
        public final int width, height, channels, depth, frames;
        public final int[] nativeLevels;

        public Info(int width, int height, int channels, int depth, int frames, int... nativeLevels) {
            if (width < 1 || height < 1 || channels < 1 || depth < 1 || frames < 1)
                throw new IllegalArgumentException("Preview dimensions must be positive.");
            this.width = width;
            this.height = height;
            this.channels = channels;
            this.depth = depth;
            this.frames = frames;
            this.nativeLevels = nativeLevels.clone();
        }
    }

    Info info();

    /**
     * Reads only this region at Z/T; level L means a 2^L downsample. Return display RGB pixels.
     * Readers should use native pyramid levels where available and keep temporary decoding bounded.
     * Stale requests are cancelled between tiles, without interrupting an active shared read.
     */
    BufferedImage read(int z, int t, int level, Rectangle region) throws IOException;

    /** Allows cooperative cancellation inside a bounded region read; null means cancelled. */
    default BufferedImage read(int z, int t, int level, Rectangle region, BooleanSupplier cancelled) throws IOException {
        return cancelled.getAsBoolean() ? null : read(z, t, level, region);
    }

    @Override void close() throws IOException;
}
