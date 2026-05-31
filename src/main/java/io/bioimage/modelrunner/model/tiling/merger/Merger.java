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
import java.util.function.Function;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Base class for merging outputs produced by tiled inference.
 * <p>
 * The base class is tensor-based so all mergers can be used by the same model
 * inference loop. Subclasses decide how patch outputs are combined inside the
 * returned tensors.
 *
 * @param <I> model input type handled by this merger
 * @param <O> model output type reconstructed by this merger
 */
public abstract class Merger<I extends Tensor<?>, O extends Tensor<?>> {

    protected static final int X1 = 0;
    protected static final int Y1 = 1;
    protected static final int X2 = 2;
    protected static final int Y2 = 3;
    protected static final int WINDOW_LENGTH = 4;

    
    protected boolean configured;
    protected boolean digested;
    private final List<Function<List<O>, List<O>>> reconstructionCallbacks =
            new ArrayList<Function<List<O>, List<O>>>();
    private boolean callbacksApplied;

    /**
     * Creates a new Merger instance.
     */
    protected Merger() {
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
     * Adds a callback that is run once on the reconstructed outputs before they
     * are returned by {@link #getReconstructed()}.
     *
     * @param callback transformation to apply to reconstructed outputs
     */
    public abstract void addCallback(Function<List<O>, List<O>> callback);

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

    /**
     * Performs register callback.
     *
     * @param callback the callback to notify.
     */
    protected void registerCallback(final Function<List<O>, List<O>> callback) {
        if (callback == null) {
            throw new IllegalArgumentException("Reconstruction callback cannot be null.");
        }
        reconstructionCallbacks.add(callback);
        callbacksApplied = false;
    }

    /**
     * Returns the result of apply reconstruction callbacks.
     *
     * @param reconstructed the reconstructed.
     * @return the resulting list.
     */
    protected List<O> applyReconstructionCallbacks(final List<O> reconstructed) {
        if (callbacksApplied || reconstructionCallbacks.isEmpty()) {
            return reconstructed;
        }
        List<O> current = reconstructed;
        for (Function<List<O>, List<O>> callback : reconstructionCallbacks) {
            current = callback.apply(current);
            if (current == null) {
                current = Collections.emptyList();
            }
        }
        callbacksApplied = true;
        return current;
    }

    /**
     * Performs reset reconstruction callbacks.
     */
    protected void resetReconstructionCallbacks() {
        callbacksApplied = false;
    }

	/**
	 * Performs require configured.
	 */
	protected void requireConfigured() {
		if (!configured) {
			throw new IllegalStateException(getClass().getSimpleName() + " must be configured before use.");
		}
	}

	/**
	 * Performs require digested.
	 */
	protected void requireDigested() {
		if (!digested) {
			throw new IllegalStateException("Merger cannot reconstruct outputs before digesting a patch.");
		}
	}

	/**
	 * Performs patch number valid.
	 *
	 * @param patchNumber the patch number.
	 */
	protected void patchNumberValid(final int patchNumber) {
		if (patchNumber < 0 || patchNumber >= this.getNPatches()) {
			throw new IllegalArgumentException("Patch should be >=0 and <" + this.getNPatches());
		}
	}

    /**
     * Returns the result of copy window.
     *
     * @param window the window.
     * @param name the name.
     * @return the resulting long.
     */
    protected static long[] copyWindow(final long[] window, final String name) {
        validateWindow(window, name);
        return window.clone();
    }

    /**
     * Performs validate window.
     *
     * @param window the window.
     * @param name the name.
     */
    protected static void validateWindow(final long[] window, final String name) {
        if (window == null || window.length < WINDOW_LENGTH) {
            throw new IllegalArgumentException(name + " must be [x1, y1, x2, y2].");
        }
        if (window[X2] <= window[X1] || window[Y2] <= window[Y1]) {
            throw new IllegalArgumentException(name + " has invalid limits: ["
                    + window[X1] + ", " + window[Y1] + ", " + window[X2] + ", " + window[Y2] + "].");
        }
    }

    /**
     * Returns the result of immutable copy.
     *
     * @param <E> the E type parameter.
     * @param values the values.
     * @return the resulting list.
     */
    protected static <E> List<E> immutableCopy(final List<E> values) {
        if (values == null) {
            return Collections.emptyList();
        }
        return Collections.unmodifiableList(new ArrayList<E>(values));
    }

    /**
     * Returns the result of find image inputs.
     *
     * @param <T> the T type parameter.
     * @param inputs the inputs to process.
     * @return the resulting list.
     */
    protected static <T extends RealType<T> & NativeType<T>> List<InputImage<T>> findImageInputs(
            final List<Tensor<T>> inputs) {
        if (inputs == null || inputs.isEmpty()) {
            return Collections.emptyList();
        }
        final List<InputImage<T>> imageInputs = new ArrayList<InputImage<T>>();
        for (Tensor<T> input : inputs) {
            if (hasXY(input)) {
                imageInputs.add(new InputImage<T>(input));
            }
        }
        return Collections.unmodifiableList(imageInputs);
    }

    /**
     * Returns whether has XY.
     *
     * @param <T> the T type parameter.
     * @param tensor the tensor.
     * @return true if has XY; false otherwise.
     */
    protected static <T extends RealType<T> & NativeType<T>> boolean hasXY(final Tensor<T> tensor) {
        return tensor != null
                && tensor.getAxesOrderString().indexOf('x') >= 0
                && tensor.getAxesOrderString().indexOf('y') >= 0;
    }

    /**
     * Returns the result of axis size.
     *
     * @param dims the dimensions.
     * @param axes the axes.
     * @param axis the axis.
     * @return the resulting long.
     */
    protected static long axisSize(final long[] dims, final String axes, final char axis) {
        if (dims == null || axes == null) {
            throw new IllegalArgumentException("Dimensions and axes cannot be null.");
        }
        final int index = axes.indexOf(axis);
        if (index < 0 || index >= dims.length) {
            throw new IllegalArgumentException("Axes '" + axes + "' do not contain axis '" + axis + "'.");
        }
        return dims[index];
    }

    /**
     * Returns the result of to XYXY window.
     *
     * @param tilePosition the tile position.
     * @param tileSize the tile size.
     * @param axes the axes.
     * @return the resulting long.
     */
    protected static long[] toXyxyWindow(final long[] tilePosition, final long[] tileSize, final String axes) {
        final int x = axes.indexOf('x');
        final int y = axes.indexOf('y');
        if (x < 0 || y < 0 || tilePosition == null || tileSize == null
                || tilePosition.length <= Math.max(x, y) || tileSize.length <= Math.max(x, y)) {
            throw new IllegalArgumentException("Tile position and size must contain x and y dimensions.");
        }
        return new long[] {
                tilePosition[x],
                tilePosition[y],
                tilePosition[x] + tileSize[x],
                tilePosition[y] + tileSize[y]
        };
    }

    /**
     * Returns the result of scale window.
     *
     * @param referenceWindow the reference window.
     * @param reference the reference.
     * @param target the target.
     * @return the resulting long.
     */
    protected static long[] scaleWindow(final long[] referenceWindow, final InputImage<?> reference,
            final InputImage<?> target) {
        if (reference == target) {
            return referenceWindow.clone();
        }
        return new long[] {
                scale(referenceWindow[X1], reference.width(), target.width()),
                scale(referenceWindow[Y1], reference.height(), target.height()),
                scale(referenceWindow[X2], reference.width(), target.width()),
                scale(referenceWindow[Y2], reference.height(), target.height())
        };
    }

    /**
     * Returns the result of scale.
     *
     * @param value the value.
     * @param sourceSize the source size.
     * @param targetSize the target size.
     * @return the resulting long.
     */
    protected static long scale(final long value, final long sourceSize, final long targetSize) {
        return clip(Math.round(value * (targetSize / (double) sourceSize)), 0L, targetSize);
    }

    /**
     * Returns the result of clip.
     *
     * @param value the value.
     * @param min the min.
     * @param max the max.
     * @return the resulting long.
     */
    protected static long clip(final long value, final long min, final long max) {
        return Math.max(min, Math.min(max, value));
    }

    protected static final class InputImage<T extends RealType<T> & NativeType<T>> {

        protected final Tensor<T> tensor;
        protected final String axes;
        protected final int xAxis;
        protected final int yAxis;
        protected final long[] dims;

        /**
         * Creates a new InputImage instance.
         *
         * @param tensor the tensor.
         */
        protected InputImage(final Tensor<T> tensor) {
            this.tensor = tensor;
            this.axes = tensor.getAxesOrderString();
            this.xAxis = axes.indexOf('x');
            this.yAxis = axes.indexOf('y');
            this.dims = tensor.getData().dimensionsAsLongArray();
        }

        /**
         * Returns the result of width.
         *
         * @return the resulting long.
         */
        protected long width() {
            return dims[xAxis];
        }

        /**
         * Returns the result of height.
         *
         * @return the resulting long.
         */
        protected long height() {
            return dims[yAxis];
        }
    }
}
