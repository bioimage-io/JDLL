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
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.compress.archivers.ArchiveException;

import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelDependencies;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat;
import io.bioimage.modelrunner.bioimageio.tiling.ImageInfo;
import io.bioimage.modelrunner.bioimageio.tiling.TileCalculator;
import io.bioimage.modelrunner.bioimageio.tiling.TileInfo;
import io.bioimage.modelrunner.bioimageio.tiling.TileMaker;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.processing.Processing;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.Constants;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;

public class BioimageIoModelPytorch extends DLModelPytorch {
	/**
	 * Object containing the information of the rdf.yaml file of a Bioimage.io model
	 */
	protected ModelDescriptor descriptor;
	/**
	 * Calculates the tile sizes depending on the model specs
	 */
	protected TileCalculator tileCalculator;
	
	protected BioimageIoModelPytorch(String modelFile, String callable, String weightsPath, 
			Map<String, Object> kwargs, ModelDescriptor descriptor) throws IOException {
		super(modelFile, callable, weightsPath, kwargs);
		this.tiling = true;
		this.descriptor = descriptor;
		this.tileCalculator = TileCalculator.init(descriptor);
	}
	
	public static BioimageIoModelPytorch create(ModelDescriptor descriptor) throws IOException {
		if (descriptor.getWeights().getModelWeights(ModelWeight.getPytorchID()) == null)
			throw new IllegalArgumentException("The model provided does not have weights in the required format, "
					+ ModelWeight.getPytorchID() + ".");
		WeightFormat pytorchWeights = descriptor.getWeights().getModelWeights(ModelWeight.getPytorchID());
		String modelFile = descriptor.getModelPath() +  File.separator + pytorchWeights.getArchitecture().getSource();
		String callable = pytorchWeights.getArchitecture().getCallable();
		String weightsFile = descriptor.getModelPath() +  File.separator + pytorchWeights.getSource();
		Map<String, Object> kwargs = pytorchWeights.getArchitecture().getKwargs();
		return new BioimageIoModelPytorch(modelFile, callable, weightsFile, kwargs, descriptor);
	}
	
	public static BioimageIoModelPytorch create(String modelPath) throws IOException {
		return create(ModelDescriptorFactory.readFromLocalFile(modelPath + File.separator + Constants.RDF_FNAME));
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
	
	/**
	 * 
	 * @param <T>
	 * 	nothing
	 * @param args
	 * 	nothing
	 * @throws IOException	nothing
	 * @throws LoadEngineException	nothing
	 * @throws RunModelException	nothing
	 * @throws LoadModelException	nothing
	 * @throws URISyntaxException 
	 * @throws ArchiveException 
	 * @throws MambaInstallException 
	 * @throws RuntimeException 
	 * @throws InterruptedException 
	 */
	public static <T extends NativeType<T> & RealType<T>> void main(String[] args) throws IOException, LoadEngineException, RunModelException, LoadModelException, InterruptedException, RuntimeException, MambaInstallException, ArchiveException, URISyntaxException {
		
		String mm = "/home/carlos/git/deepimagej-plugin/models/OC1 Project 11 Cellpose_24022025_131039";
		Img<T> im = Cast.unchecked(ArrayImgs.floats(new long[] {1, 1, 1024, 1024}));
		List<Tensor<T>> l = new ArrayList<Tensor<T>>();
		l.add(Tensor.build("input", "bcyx", im));
		//BioimageIoModelPytorch.installRequirements();
		BioimageIoModelPytorch model = create(mm);
		List<String> missing = model.findMissingDependencies();
		model.loadModel();
		TileInfo tile = TileInfo.build(l.get(0).getName(), new long[] {1, 1, 512, 512}, 
				l.get(0).getAxesOrderString(), new long[] {1, 1, 512, 512}, l.get(0).getAxesOrderString());
		List<TileInfo> tileList = new ArrayList<TileInfo>();
		tileList.add(tile);
		model.run(l);
		System.out.println(false);
		
	}

}
