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

import java.util.Collections;
import java.util.List;
import java.util.function.Function;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Pass-through merger for models that do not require tiled inference.
 * <p>
 * A no-tile inference run should produce exactly one output tensor for the
 * complete input. This merger returns that tensor unchanged.
 */
public final class NoTileMerger<T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
		extends Merger<Tensor<T>, Tensor<R>> {

	private List<Tensor<T>> inputs = Collections.emptyList();
	private List<Tensor<R>> reconstructed = Collections.emptyList();

	public NoTileMerger() {
		super();
	}

	@Override
	public void configure(final List<Tensor<T>> inputs) {
		this.inputs = inputs;
		this.configured = true;
		this.digested = false;
	}

	@Override
	public List<Tensor<T>> get(final int patchNumber) {
		requireConfigured();
		patchNumberValid(patchNumber);
		return inputs;
	}

	@Override
	public int getNPatches() {
		return 1;
	}

	@Override
	public void digest(final int patchNumber, final List<Tensor<R>> outputs) {
		requireConfigured();
		patchNumberValid(patchNumber);
		reconstructed = outputs;
		digested = true;
		resetReconstructionCallbacks();
	}

	@Override
	public void addCallback(final Function<List<Tensor<R>>, List<Tensor<R>>> callback) {
		registerCallback(callback);
	}

	@Override
	public List<Tensor<R>> getReconstructed() {
		requireConfigured();
		requireDigested();
		reconstructed = applyReconstructionCallbacks(reconstructed);
		return reconstructed;
	}
}
