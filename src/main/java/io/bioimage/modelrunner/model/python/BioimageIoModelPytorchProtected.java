/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;


import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelDependencies;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.bioimageio.tiling.ImageInfo;
import io.bioimage.modelrunner.bioimageio.tiling.TileCalculator;
import io.bioimage.modelrunner.bioimageio.tiling.TileInfo;
import io.bioimage.modelrunner.bioimageio.tiling.TileMaker;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.processing.Processing;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class BioimageIoModelPytorchProtected extends DLModelPytorchProtected {
	/**
	 * Object containing the information of the rdf.yaml file of a Bioimage.io model
	 */
	protected ModelDescriptor descriptor;
	/**
	 * Calculates the tile sizes depending on the model specs
	 */
	protected TileCalculator tileCalculator;
	
	protected BioimageIoModelPytorchProtected(String modelFile, String callable, String weightsPath, 
			Map<String, Object> kwargs, ModelDescriptor descriptor, boolean custom) throws IOException {
		super(modelFile, callable, weightsPath, kwargs, custom);
		this.tiling = true;
		this.descriptor = descriptor;
		this.tileCalculator = TileCalculator.init(descriptor);
	}
		
	protected BioimageIoModelPytorchProtected(String modelFile, String callable, String weightsPath, 
			Map<String, Object> kwargs, ModelDescriptor descriptor) throws IOException {
		this(modelFile, callable, weightsPath, kwargs, descriptor, false);
	}
	
	/**
	 * Run a Bioimage.io model and execute the tiling strategy in one go.
	 * The model needs to have been previously loaded with {@link #loadModel()}.
	 * This method does not execute pre- or post-processing, they
	 * need to be executed independently before or after
	 * 
	 * @param <T>
	 * 	ImgLib2 data type of the output images
	 * @param <R>
	 * 	ImgLib2 data type of the input images
	 * @param inputTensors
	 * 	list of the input tensors that are going to be inputed to the model
	 * @return the resulting tensors 
	 * @throws RunModelException if the model has not been previously loaded
	 * @throws IllegalArgumentException if the model is not a Bioimage.io model or if lacks a Bioimage.io
	 *  rdf.yaml specs file in the model folder. 
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> run(List<Tensor<R>> inputTensors) throws RunModelException {
		if (!this.isLoaded())
			throw new RunModelException("Please first load the model.");
		if (!this.tiling) {
			List<Tensor<T>> outs = createOutputTensors();
			this.runNoTiles(inputTensors, outs);
			return outs;
		}
		List<ImageInfo> imageInfos = inputTensors.stream()
				.map(tt -> new ImageInfo(tt.getName(), tt.getAxesOrderString(), tt.getData().dimensionsAsLongArray()))
				.collect(Collectors.toList());
		List<TileInfo> inputTiles = tileCalculator.getOptimalTileSize(imageInfos);
		TileMaker maker = TileMaker.build(descriptor, inputTiles);
		List<Tensor<T>> outTensors = createOutputTensors(maker);
		return runBMZ(inputTensors, outTensors, maker);
	}
	
	private <T extends RealType<T> & NativeType<T>> List<Tensor<T>> createOutputTensors(TileMaker maker) {
		List<Tensor<T>> outputTensors = new ArrayList<Tensor<T>>();
		for (TensorSpec tt : descriptor.getOutputTensors()) {
			long[] dims = maker.getOutputImageSize(tt.getName());
			outputTensors.add((Tensor<T>) Tensor.buildBlankTensor(tt.getName(), 
																	tt.getAxesOrder(), 
																	dims, 
																	(T) new FloatType()));
		}
		return outputTensors;
	}
	
	private <T extends RealType<T> & NativeType<T>> List<Tensor<T>> createOutputTensors() {
		List<Tensor<T>> outputTensors = new ArrayList<Tensor<T>>();
		for (TensorSpec tt : descriptor.getOutputTensors()) {
			outputTensors.add(Tensor.buildEmptyTensor(tt.getName(), tt.getAxesOrder()));
		}
		return outputTensors;
	}
	
	private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> runBMZ(List<Tensor<R>> inputTensors, List<Tensor<T>> outputTensors, TileMaker tiles) throws RunModelException {
		Processing processing = Processing.init(descriptor);
		inputTensors = processing.preprocess(inputTensors, false);
		runTiling(inputTensors, outputTensors, tiles);
		return processing.postprocess(outputTensors, true);
	}

	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	void run(List<Tensor<T>> inputTensors, List<Tensor<R>> outputTensors) throws RunModelException {
		if (!this.isLoaded())
			throw new RunModelException("Please first load the model.");
		if (!this.tiling) {
			this.runNoTiles(inputTensors, outputTensors);
			return;
		}
		List<ImageInfo> imageInfos = inputTensors.stream()
				.map(tt -> new ImageInfo(tt.getName(), tt.getAxesOrderString(), tt.getData().dimensionsAsLongArray()))
				.collect(Collectors.toList());
		List<TileInfo> inputTiles = tileCalculator.getOptimalTileSize(imageInfos);
		TileMaker maker = TileMaker.build(descriptor, inputTiles);
		for (int i = 0; i < maker.getNumberOfTiles(); i ++) {
			Tensor<R> tt = outputTensors.get(i);
			long[] expectedSize = maker.getOutputImageSize(tt.getName());
			if (expectedSize == null) {
				throw new IllegalArgumentException("Tensor '" + tt.getName() + "' is missing in the outputs.");
			} else if (!tt.isEmpty() && Arrays.equals(expectedSize, tt.getData().dimensionsAsLongArray())) {
				throw new IllegalArgumentException("Tensor '" + tt.getName() + "' size is different than the expected size"
						+ " as defined by the rdf.yaml: " + Arrays.toString(tt.getData().dimensionsAsLongArray()) 
						+ " vs " + Arrays.toString(expectedSize) + ".");
			}
		}
		runBMZ(inputTensors, outputTensors, maker);
	}
	
	public List<String> findMissingDependencies() {
		Mamba mamba = new Mamba(new File(envPath).getParentFile().getParentFile().getAbsolutePath());
		List<String> reqDeps = ModelDependencies.getDependencies(descriptor, 
				descriptor.getWeights().getModelWeights(ModelWeight.getPytorchID()));
		try {
			return mamba.checkUninstalledDependenciesInEnv(this.envPath, reqDeps);
		} catch (MambaInstallException e) {
			return reqDeps;
		}
	}
	
	public boolean allDependenciesInstalled() {
		return findMissingDependencies().size() == 0;
	}

}
