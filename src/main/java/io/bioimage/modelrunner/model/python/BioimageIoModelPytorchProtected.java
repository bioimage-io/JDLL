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
	 * @param modelFile the model file.
	 * @param callable the callable.
	 * @param importModule the import module.
	 * @param weightsPath the weights path.
	 * @param kwargs the kwargs.
	 * @param descriptor the descriptor.
	 * @param custom the custom.
	 * @param device the device.
	 * @throws BuildException if the Python environment or service cannot be built.
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
	 * @param modelFile the model file.
	 * @param callable the callable.
	 * @param importModule the import module.
	 * @param weightsPath the weights path.
	 * @param kwargs the kwargs.
	 * @param descriptor the descriptor.
	 * @param device the device.
	 * @throws BuildException if the Python environment or service cannot be built.
	 */
	protected BioimageIoModelPytorchProtected(String modelFile, String callable, String importModule, String weightsPath, 
			Map<String, Object> kwargs, ModelDescriptor descriptor, String device) throws BuildException {
		this(modelFile, callable, importModule, weightsPath, kwargs, descriptor, false, device);
	}

	/**
	 * Returns the output tensor axes.
	 *
	 * @param outputCount the output count.
	 * @return the output tensor axes.
	 */
	@Override
	protected String getOutputTensorAxes(int outputCount) {
		if (descriptor.getOutputTensors().size() <= outputCount)
			throw new IllegalArgumentException("Cellpose only has 6 outputs.");
		return this.descriptor.getOutputTensors().get(outputCount).getAxesOrder();
	}

	/**
	 * Returns the tile maker.
	 *
	 * @param <T> the T type parameter.
	 * @param <R> the R type parameter.
	 * @param inputs the inputs to process.
	 * @return the tile maker.
	 */
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
