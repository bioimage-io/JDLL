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
package io.bioimage.modelrunner.model;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

import io.bioimage.modelrunner.bioimageio.tiling.TileInfo;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.java.DLModelJava.TilingConsumer;
import io.bioimage.modelrunner.model.tiling.merger.Merger;
import io.bioimage.modelrunner.model.tiling.merger.NoTileMerger;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Class that manages a Deep Learning model to load it and run it.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public abstract class BaseModel implements Closeable
{
	/**
	 * Whether the model is loaded or not
	 */
	protected boolean loaded = false;
	/**
	 * Whether the model is closed or not
	 */
	protected boolean closed = false;

	/**
	 * Path to the folder containing the Bioimage.io model
	 */
	protected String modelFolder;

    protected volatile boolean inferenceCancellationRequested;

    /**
     * List containing the desired tiling strategy for each of the input tensors.
     */
    protected List<TileInfo> inputTiles;

    /**
     * List containing the desired tiling strategy for each of the output tensors.
     */
    protected List<TileInfo> outputTiles;

    /**
     * Consumer used to inform the current tile being processed and in how many
     * tiles the input images are going to be separated.
     */
    protected TilingConsumer tileCounter;

    /**
     * Consumer used to report structured inference progress events.
     */
    protected Consumer<InferenceProgress> inferenceProgressConsumer;

	/**
	 * Load the model wanted to make inference into the particular ClassLoader
	 * created to run a specific Deep Learning framework (engine)
	 * 
	 * @throws LoadModelException
	 *             if the model was not loaded
	 */
	public abstract void loadModel() throws LoadModelException;

	/**
	 * Close the Deep LEarning model in the ClassLoader where the Deep Learning
	 * framework has been called and instantiated
	 */
	@Override
	public abstract void close();
    
    protected abstract <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> backboneSingleInferenceTile(final List<Tensor<T>> inputs) throws RunModelException;

	/**
	 * Get the folder where this model is located
	 * 
	 * @return the folder where this model is located
	 */
	public String getModelFolder()
	{
		return this.modelFolder;
	}
	
	/**
	 * Whether the model is loaded or not
	 * @return whether the model is loaded or not
	 */
	public boolean isLoaded() {
		return loaded;
	}
	
	/**
	 * Whether the model is closed or not
	 * @return whether the model is closed or not
	 */
	public boolean isClosed() {
		return closed;
	}

    /**
     * Sets a consumer used to track tile execution progress.
     *
     * @param tileCounter
     *     consumer used to track tile inference
     */
    public void setTilingCounter(final TilingConsumer tileCounter) {
        this.tileCounter = tileCounter;
    }
    
    /**
     * 
     * @return an object that can be used to track the progress processing tiles
     */
    public TilingConsumer getTilingCounter() {
    	return this.tileCounter;
    }

    /**
     * Sets a consumer used to receive structured inference progress events.
     *
     * @param consumer
     *     consumer called as inference advances
     */
    public void setInferenceProgressConsumer(final Consumer<InferenceProgress> consumer) {
        this.inferenceProgressConsumer = consumer;
    }

    /**
     * @return the configured structured inference progress consumer, or null
     */
    public Consumer<InferenceProgress> getInferenceProgressConsumer() {
        return inferenceProgressConsumer;
    }

    /**
     * Runs inference directly on a list of input images.
     *
     * @param <T> input data type
     * @param <R> output data type
     * @param inputs input images
     * @return output images
     * @throws RunModelException if model execution fails
     */
    public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> inference(final Tensor<T>... inputs) throws RunModelException {
        inferenceCancellationRequested = false;
    	List<List<Tensor<T>>> inputBatches = new ArrayList<List<Tensor<T>>>();
    	for (Tensor<T> inp : inputs) {
    		List<Tensor<T>> inpList = new ArrayList<Tensor<T>>();
    		inpList.add(inp);
    		inputBatches.add(inpList);
    	}
    	List<List<Tensor<R>>> batchedOuts = backboneBatchInference(inputBatches);
    	List<Tensor<R>> singleOutput = new ArrayList<>();
    	for (List<Tensor<R>>batched : batchedOuts) {
    		singleOutput.add(batched.get(0));
    	}
    	return singleOutput;
    }

    protected String getOutputTensorAxes(int outputCount) {
		return "bcyx";
	}

    /**
     * Emits a structured inference progress event. Progress reporting must not
     * affect inference execution.
     *
     * @param progress
     *     progress event
     */
    protected void emitProgress(final InferenceProgress progress) {
        if (inferenceProgressConsumer == null || progress == null) {
            return;
        }
        try {
            inferenceProgressConsumer.accept(progress);
        } catch (RuntimeException e) {
            // Progress listeners are observational and should not break model execution.
        }
    }

    protected void throwIfInferenceCancelled() throws RunModelException {
        if (inferenceCancellationRequested || Thread.currentThread().isInterrupted()) {
            throw new RunModelException("Inference cancelled.");
        }
    }

    /**
     * Runs inference directly on a list of input images.
     *
     * @param <T> input data type
     * @param <R> output data type
     * @param inputs input images
     * @return output images
     * @throws RunModelException if model execution fails
     */
    public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<List<Tensor<R>>> inferenceBatch(final List<Tensor<T>>... batchedInputs)
            throws RunModelException {
        inferenceCancellationRequested = false;
    	return backboneBatchInference(Arrays.asList(batchedInputs));
    }
    
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<List<Tensor<R>>> backboneBatchInference(final List<List<Tensor<T>>> batchedInputs) throws RunModelException {

        if (!loaded) {
            throw new RuntimeException("Please load the model first.");
        }
        int batchInd = 0;
        int imsInBatch = batchedInputs.get(0).size();
        for (List<Tensor<T>> batch : batchedInputs) {
        	if (batch.size() != imsInBatch)
        		throw new IllegalArgumentException("All batches must have the same number of tensors. Batch of input 0 has "
        				+ imsInBatch +  " and batch of input " + batchInd + " has " + batch.size());
        	String name = batch.get(0).getName();
        	for (int i = 1; i < batch.size(); i ++) {
        		String nName = batch.get(i).getName();
        		if (nName.equals(name))
        			throw new IllegalArgumentException("All tensors of a batch should be named equally. For batch " + batchInd
        					+ "at least two different names where found: " + name + " (pos 0) and " + nName + " (pos" + i + ")");
        	}
        	batchInd ++;
        }
        
        List<List<Tensor<R>>> outputs = new ArrayList<List<Tensor<R>>>();
        for (int i = 0; i < batchedInputs.get(0).size(); i ++) {
            throwIfInferenceCancelled();
        	List<Tensor<T>> inputs = new ArrayList<Tensor<T>>();
        	for (int j = 0; j < batchedInputs.size(); j ++) {
        		inputs.add(batchedInputs.get(j).get(i));
        	}
        	List<Tensor<R>> aa = backboneSingleInference(inputs);
        	for (int k = 0; k < aa.size(); k ++) {
        		if (outputs.size() < k +1)
        			outputs.add(new ArrayList<Tensor<R>>());
        		outputs.get(k).add(aa.get(k));
        	}
        }
        return outputs;
    }
    
    protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    Merger<Tensor<T>, Tensor<R>> getTileMaker(final List<Tensor<T>> inputs) {
        Merger<Tensor<T>, Tensor<R>> merger = new NoTileMerger<T, R>();
        merger.configure(inputs);
        return merger;
    }
    
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> backboneSingleInference(final List<Tensor<T>> inputs) throws RunModelException {

        throwIfInferenceCancelled();
        Merger<Tensor<T>, Tensor<R>> merger = getTileMaker(inputs);
        if (tileCounter == null) {
            tileCounter = new TilingConsumer();
        }

        int nPatches = merger.getNPatches();
        tileCounter.acceptTotal((long) nPatches);
        emitProgress(InferenceProgress.inferenceStart(nPatches));
        for (int i = 0; i < nPatches; i ++) {
            throwIfInferenceCancelled();
            emitProgress(InferenceProgress.patchStart(i + 1, nPatches));
            List<Tensor<R>> tiledOutputs = backboneSingleInferenceTile(merger.get(i));
            throwIfInferenceCancelled();
            merger.digest(i, tiledOutputs);
            tileCounter.acceptProgress((long) (i + 1));
            emitProgress(InferenceProgress.patchEnd(i + 1, nPatches));
        }
        throwIfInferenceCancelled();
        emitProgress(InferenceProgress.mergeStart());
        List<Tensor<R>> reconstructed = merger.getReconstructed();
        throwIfInferenceCancelled();
        emitProgress(InferenceProgress.inferenceEnd());
        return reconstructed;
    }

    protected List<String> getOutputAxes(final int outputCount) {
        final List<String> axes = new ArrayList<String>(outputCount);
        for (int i = 0; i < outputCount; i ++) {
            axes.add(getOutputTensorAxes(i));
        }
        return axes;
    }
}
