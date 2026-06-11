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

import io.bioimage.modelrunner.model.tiling.TileMaker;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Merges dense tensor outputs produced from spatial tiles.
 * <p>
 * Each patch output is copied into the corresponding region of a preallocated
 * full-size output tensor.
 */
public final class DenseMerger<T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
extends Merger<Tensor<T>, Tensor<R>> {
	
	private final TileMaker tileMaker;

    private List<Tensor<T>> inputs = Collections.emptyList();
    private List<InputImage<T>> imageInputs = Collections.emptyList();
    private List<long[]> referenceWindows = Collections.emptyList();
    private List<Tensor<R>> reconstructed = Collections.emptyList();

    /**
     * Creates a new DenseMerger instance.
     *
     * @param tileMaker the tile maker.
     */
    public DenseMerger(final TileMaker tileMaker) {
        if (tileMaker == null) {
            throw new IllegalArgumentException("TileMaker cannot be null.");
        }
        this.tileMaker = tileMaker;
    }

    /**
     * Performs configure.
     *
     * @param inputs the inputs to process.
     */
    @Override
    public void configure(final List<Tensor<T>> inputs) {
        this.inputs = inputs == null ? Collections.<Tensor<T>>emptyList() : inputs;
        this.imageInputs = findImageInputs(this.inputs);
        if (imageInputs.isEmpty()) {
            throw new IllegalArgumentException("DenseMerger needs at least one input tensor with x and y axes.");
        }
        final InputImage<T> reference = imageInputs.get(0);
        this.referenceWindows = createReferenceWindows(reference);

        this.reconstructed = tileMaker.createOutputTensors((R) new net.imglib2.type.numeric.real.FloatType(0.0f));
        this.configured = true;
        this.digested = referenceWindows.isEmpty();
        resetReconstructionCallbacks();
    }

	/**
	 * Returns the result of get.
	 *
	 * @param patchNumber the patch number.
	 * @return the resulting list.
	 */
	@Override
	public List<Tensor<T>> get(int patchNumber) {
        requireConfigured();
        patchNumberValid(patchNumber);
        return getTileMakerPatch(patchNumber);
	}

	/**
	 * Returns the output.
	 *
	 * @param patchNumber the patch number.
	 * @return the output.
	 */
	public List<Tensor<R>> getOutput(int patchNumber) {
		List<Tensor<R>> outTiles = new ArrayList<>();
		for (Tensor<R> tt : this.reconstructed) {
			 Tensor<R> target = tileMaker.getNthTileOutput(tt, patchNumber);
			 outTiles.add(target);
		}
        return outTiles;
	}

	/**
	 * Returns the n patches.
	 *
	 * @return the n patches.
	 */
	@Override
	public int getNPatches() {
        requireConfigured();
        return referenceWindows.size();
	}

    /**
     * Performs digest.
     *
     * @param patchNumber the patch number.
     * @param outputs the outputs to populate.
     */
    @Override
    public void digest(final int patchNumber, final List<Tensor<R>> outputs) {
        requireConfigured();
        patchNumberValid(patchNumber);
        if (outputs == null || outputs.size() != reconstructed.size()) {
            throw new IllegalArgumentException("DenseMerger expected " + reconstructed.size()
                    + " output tensor(s), got " + (outputs == null ? 0 : outputs.size()) + ".");
        }
        for (int i = 0; i < outputs.size(); i++) {
            copyPatchIntoReconstruction(outputs.get(i), reconstructed.get(i), patchNumber);
        }
        digested = patchNumber + 1 == this.getNPatches();
        resetReconstructionCallbacks();
    }

    /**
     * Performs add callback.
     *
     * @param callback the callback to notify.
     */
    @Override
    public void addCallback(final Function<List<Tensor<R>>, List<Tensor<R>>> callback) {
        registerCallback(callback);
    }

    /**
     * Returns the reconstructed.
     *
     * @return the reconstructed.
     */
    @Override
    public List<Tensor<R>> getReconstructed() {
        requireConfigured();
        requireDigested();
        reconstructed = applyReconstructionCallbacks(reconstructed);
        return reconstructed;
    }

    private void copyPatchIntoReconstruction(final Tensor<R> patchOutput,
            final Tensor<R> fullOutput, final int patchNumber) {
        if (patchOutput == null || fullOutput == null) {
            throw new IllegalArgumentException("DenseMerger cannot digest null output tensors.");
        }
        final RandomAccessibleInterval<R> source = patchOutput.getData();
        final RandomAccessibleInterval<R> target = tileMaker.getNthTileOutput(fullOutput, patchNumber).getData();
        LoopBuilder.setImages(source, target)
                .multiThreaded()
                .forEachPixel((src, dst) -> dst.set(src));
    }

    private List<long[]> createReferenceWindows(final InputImage<T> reference) {
        return createTileMakerWindows(reference);
    }

    private List<long[]> createTileMakerWindows(final InputImage<T> reference) {
        final List<long[]> positions = tileMaker.getTilePostionsInputImage(reference.tensor.getName());
        final long[] tileSize = tileMaker.getInputTileSize(reference.tensor.getName());
        final List<long[]> windows = new ArrayList<long[]>(positions.size());
        for (long[] position : positions) {
            windows.add(toXyxyWindow(position, tileSize, reference.axes));
        }
        return Collections.unmodifiableList(windows);
    }

    private List<Tensor<T>> getTileMakerPatch(final int patchNumber) {
        final List<Tensor<T>> patch = new ArrayList<Tensor<T>>(inputs.size());
        for (Tensor<T> input : inputs) {
            if (hasXY(input)) {
                patch.add(tileMaker.getNthTileInput(input, patchNumber));
            } else {
                patch.add(input);
            }
        }
        return patch;
    }

}
