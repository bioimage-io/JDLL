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
package io.bioimage.modelrunner.bioimageio.tiling.merger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Base class for merging outputs produced by tiled inference.
 * <p>
 * The base class deliberately does not use ImgLib2 generic bounds. A merger can
 * combine detections, dense tensors, classifications or any future semantic
 * output. Subclasses decide the tile output type {@code I} and final merged
 * output type {@code O}.
 *
 * @param <I> per-tile output type accepted by this merger
 * @param <O> final merged output type returned by this merger
 */
public abstract class Merger<I, O> {

    protected static final int X1 = 0;
    protected static final int Y1 = 1;
    protected static final int X2 = 2;
    protected static final int Y2 = 3;
    protected static final int WINDOW_LENGTH = 4;

    private final long[] outputWindow;
    private final List<TileOutput<I>> tileOutputs = new ArrayList<TileOutput<I>>();

    protected Merger(final long[] outputWindow) {
        this.outputWindow = copyWindow(outputWindow, "Output window");
    }

    public final void addTileOutput(final I output, final long[] tileWindow) {
        tileOutputs.add(new TileOutput<I>(output, tileWindow));
    }

    public final O merge() {
        return merge(Collections.unmodifiableList(tileOutputs));
    }

    protected abstract O merge(List<TileOutput<I>> tileOutputs);

    protected final long[] getOutputWindow() {
        return outputWindow.clone();
    }

    protected final long outputWidth() {
        return outputWindow[X2] - outputWindow[X1];
    }

    protected final long outputHeight() {
        return outputWindow[Y2] - outputWindow[Y1];
    }

    protected final long tileOffsetX(final long[] tileWindow) {
        validateWindow(tileWindow, "Tile window");
        return tileWindow[X1] - outputWindow[X1];
    }

    protected final long tileOffsetY(final long[] tileWindow) {
        validateWindow(tileWindow, "Tile window");
        return tileWindow[Y1] - outputWindow[Y1];
    }

    protected static long[] copyWindow(final long[] window, final String name) {
        validateWindow(window, name);
        return window.clone();
    }

    protected static void validateWindow(final long[] window, final String name) {
        if (window == null || window.length < WINDOW_LENGTH) {
            throw new IllegalArgumentException(name + " must be [x1, y1, x2, y2].");
        }
        if (window[X2] <= window[X1] || window[Y2] <= window[Y1]) {
            throw new IllegalArgumentException(name + " has invalid limits: ["
                    + window[X1] + ", " + window[Y1] + ", " + window[X2] + ", " + window[Y2] + "].");
        }
    }

    protected static double clip(final double value, final double min, final double max) {
        return Math.max(min, Math.min(max, value));
    }

    protected static final class TileOutput<I> {

        private final I output;
        private final long[] tileWindow;

        private TileOutput(final I output, final long[] tileWindow) {
            this.output = output;
            this.tileWindow = copyWindow(tileWindow, "Tile window");
        }

        public I getOutput() {
            return output;
        }

        public long[] getTileWindow() {
            return tileWindow.clone();
        }
    }
}
