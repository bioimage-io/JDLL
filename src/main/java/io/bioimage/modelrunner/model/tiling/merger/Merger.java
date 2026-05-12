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
package io.bioimage.modelrunner.model.tiling.merger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Base class for merging outputs produced by tiled inference.
 * <p>
 * The base class deliberately does not use ImgLib2 generic bounds. A merger can
 * prepare input patches and reconstruct detections, dense tensors,
 * classifications or any future semantic output. Subclasses decide the model
 * input type {@code I} and output type {@code O}.
 *
 * @param <I> model input type handled by this merger
 * @param <O> model output type reconstructed by this merger
 */
public abstract class Merger<I, O> {

    protected static final int X1 = 0;
    protected static final int Y1 = 1;
    protected static final int X2 = 2;
    protected static final int Y2 = 3;
    protected static final int WINDOW_LENGTH = 4;

    
    protected boolean configured;
    protected boolean digested;
    private final long[] outputWindow;

    protected Merger() {
        this.outputWindow = null;
    }

    protected Merger(final long[] outputWindow) {
        this.outputWindow = copyWindow(outputWindow, "Output window");
    }

    /**
     * Configures this merger for one inference item.
     * <p>
     * For single-input models this list contains one object. For multi-input
     * models it contains all objects that must be patched together for each
     * model invocation.
     *
     * @param inputs source objects to patch and later reconstruct
     */
    public abstract void configure(List<I> inputs);

    /**
     * Returns the model inputs for one patch.
     *
     * @param patchNumber zero-based patch index
     * @return patched input objects to pass to the model
     */
    public abstract List<I> get(int patchNumber);

    /**
     * Returns the number of model invocations needed to reconstruct the output.
     *
     * @return number of patches to process
     */
    public abstract int getNPatches();

    /**
     * Incorporates the outputs produced by one model invocation.
     *
     * @param patchNumber zero-based patch index that produced {@code outputs}
     * @param outputs model outputs for this patch
     */
    public abstract void digest(int patchNumber, List<O> outputs);

    /**
     * Convenience overload for single-output models.
     *
     * @param patchNumber zero-based patch index that produced {@code output}
     * @param output model output for this patch
     */
    public final void digest(final int patchNumber, final O output) {
        digest(patchNumber, Collections.singletonList(output));
    }

    /**
     * Returns the reconstructed outputs after all patch outputs have been
     * digested.
     *
     * @return reconstructed model outputs
     */
    public abstract List<O> getReconstructed();

	protected void requireConfigured() {
		if (!configured) {
			throw new IllegalStateException(getClass().getSimpleName() + " must be configured before use.");
		}
	}

	protected void requireDigested() {
		if (!digested) {
			throw new IllegalStateException("Merger cannot reconstruct outputs before digesting a patch.");
		}
	}

	protected void patchNumberValid(final int patchNumber) {
		if (patchNumber < 0 || patchNumber >= this.getNPatches()) {
			throw new IllegalArgumentException(String.format("Patch should be >=0 and <%s", "" + this.getNPatches()));
		}
	}

    protected final long[] getOutputWindow() {
        requireOutputWindow();
        return outputWindow.clone();
    }

    protected final long outputWidth() {
        requireOutputWindow();
        return outputWindow[X2] - outputWindow[X1];
    }

    protected final long outputHeight() {
        requireOutputWindow();
        return outputWindow[Y2] - outputWindow[Y1];
    }

    protected final long tileOffsetX(final long[] tileWindow) {
        requireOutputWindow();
        validateWindow(tileWindow, "Tile window");
        return tileWindow[X1] - outputWindow[X1];
    }

    protected final long tileOffsetY(final long[] tileWindow) {
        requireOutputWindow();
        validateWindow(tileWindow, "Tile window");
        return tileWindow[Y1] - outputWindow[Y1];
    }

    private void requireOutputWindow() {
        if (outputWindow == null) {
            throw new IllegalStateException(getClass().getSimpleName() + " was not configured with an output window.");
        }
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

    protected static <E> List<E> immutableCopy(final List<E> values) {
        if (values == null) {
            return Collections.emptyList();
        }
        return Collections.unmodifiableList(new ArrayList<E>(values));
    }
}
