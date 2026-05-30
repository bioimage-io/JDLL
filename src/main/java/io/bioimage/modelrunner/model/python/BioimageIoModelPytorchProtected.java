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
/**
 * 
 */
package io.bioimage.modelrunner.model.python;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.model.processing.Processing;
import io.bioimage.modelrunner.model.tiling.ImageInfo;
import io.bioimage.modelrunner.model.tiling.TileCalculator;
import io.bioimage.modelrunner.model.tiling.TileInfo;
import io.bioimage.modelrunner.model.tiling.TileMaker;
import io.bioimage.modelrunner.model.tiling.merger.DenseMerger;
import io.bioimage.modelrunner.model.tiling.merger.Merger;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class BioimageIoModelPytorchProtected extends DLModelPytorchProtected {
	/**
	 * Object containing the information of the rdf.yaml file of a Bioimage.io model
	 */
	protected ModelDescriptor descriptor;
	/**
	 * Calculates the tile sizes depending on the model specs
	 */
	protected TileCalculator tileCalculator;
	
	/**
	 * Creates a new BioimageIoModelPytorchProtected.
	 *
	 * @param modelFile the modelFile parameter.
	 * @param callable the callable parameter.
	 * @param importModule the importModule parameter.
	 * @param weightsPath the weightsPath parameter.
	 * @param kwargs the kwargs parameter.
	 * @param descriptor the descriptor parameter.
	 * @param custom the custom parameter.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if there is any error building the environment
	 */
	protected BioimageIoModelPytorchProtected(String modelFile, String callable, String importModule, String weightsPath, 
			Map<String, Object> kwargs, ModelDescriptor descriptor, boolean custom, String device) throws BuildException {
		super(modelFile, callable, importModule, weightsPath, kwargs, custom, device);
		this.descriptor = descriptor;
		this.tileCalculator = TileCalculator.init(descriptor);
	}
		
	/**
	 * Creates a new BioimageIoModelPytorchProtected.
	 *
	 * @param modelFile the modelFile parameter.
	 * @param callable the callable parameter.
	 * @param importModule the importModule parameter.
	 * @param weightsPath the weightsPath parameter.
	 * @param kwargs the kwargs parameter.
	 * @param descriptor the descriptor parameter.
	 * @throws BuildException if there is any error building the environmnet
	 */
	protected BioimageIoModelPytorchProtected(String modelFile, String callable, String importModule, String weightsPath, 
			Map<String, Object> kwargs, ModelDescriptor descriptor, String device) throws BuildException {
		this(modelFile, callable, importModule, weightsPath, kwargs, descriptor, false, device);
	}

	@Override
	protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	Merger<Tensor<T>, Tensor<R>> getTileMaker(final List<Tensor<T>> inputs) {

		List<ImageInfo> imageInfos = inputs.stream()
				.map(tt -> new ImageInfo(tt.getName(), tt.getAxesOrderString(), tt.getData().dimensionsAsLongArray()))
				.collect(Collectors.toList());
		this.tileCalculator = TileCalculator.init(descriptor);
		List<TileInfo> inputTiles = tileCalculator.getOptimalTileSize(imageInfos);
		TileMaker tileMaker = TileMaker.build(descriptor, inputTiles);		
		DenseMerger<T, R> merger = new DenseMerger<T, R>(tileMaker);

		Processing processing = Processing.init(descriptor);
		processing.preprocess(inputs, true);		
		
		merger.addCallback(reconstructed -> processing.postprocess(reconstructed, true));
		merger.configure(inputs);
		return merger;
	}
}
