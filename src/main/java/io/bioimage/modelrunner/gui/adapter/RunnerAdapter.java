package io.bioimage.modelrunner.gui.adapter;

import java.io.Closeable;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.bioimageio.description.exceptions.ModelSpecsException;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.BaseModel;
import io.bioimage.modelrunner.model.java.BioimageIoModelJava;
import io.bioimage.modelrunner.model.python.BioimageIoModelPytorch;
import io.bioimage.modelrunner.model.special.stardist.Stardist2D;
import io.bioimage.modelrunner.model.special.stardist.StardistAbstract;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public abstract class RunnerAdapter implements Closeable {

	protected final ModelDescriptor descriptor;
	
	protected final BaseModel model;
		
	protected boolean closed = false;
	
	protected boolean loaded = false;

	
	protected abstract <T extends RealType<T> & NativeType<T>>
	LinkedHashMap<TensorSpec, RandomAccessibleInterval<T>> displayTestInputs(LinkedHashMap<TensorSpec, String> testInputs);
	
	protected abstract LinkedHashMap<TensorSpec, String> getTestInputs();

	protected RunnerAdapter(ModelDescriptor descriptor) throws IOException, LoadEngineException {
		this.descriptor = descriptor;
		if (descriptor.getModelFamily().equals(ModelDescriptor.STARDIST)) {
			model = StardistAbstract.fromBioimageioModel(descriptor);
		} else if (descriptor.getModelFamily().equals(ModelDescriptor.BIOIMAGEIO)
				&& !descriptor.getWeights().getAllSuportedWeightNames().contains(ModelWeight.getPytorchID())) {
			model = BioimageIoModelJava.createBioimageioModel(descriptor.getModelPath());
		} else if (descriptor.getModelFamily().equals(ModelDescriptor.BIOIMAGEIO)) {
			model = BioimageIoModelPytorch.create(descriptor);
		} else {
			throw new IllegalArgumentException("Model not supported");
		}
	}

	protected RunnerAdapter(ModelDescriptor descriptor, ClassLoader classloader) throws IOException, LoadEngineException {
		this.descriptor = descriptor;
		if (descriptor.getModelFamily().equals(ModelDescriptor.STARDIST)) {
			model = StardistAbstract.fromBioimageioModel(descriptor);
		} else if (descriptor.getModelFamily().equals(ModelDescriptor.BIOIMAGEIO)
				&& !descriptor.getWeights().getAllSuportedWeightNames().contains(ModelWeight.getPytorchID())) {
			model = BioimageIoModelJava.createBioimageioModel(descriptor.getModelPath(), classloader);
		} else if (descriptor.getModelFamily().equals(ModelDescriptor.BIOIMAGEIO)) {
			model = BioimageIoModelPytorch.create(descriptor);
		} else {
			throw new IllegalArgumentException("Model not supported");
		}
	}

	protected RunnerAdapter(ModelDescriptor descriptor, String enginesPath) 
			throws IOException, LoadEngineException {
		this.descriptor = descriptor;
		if (descriptor.getModelFamily().equals(ModelDescriptor.STARDIST)) {
			model = Stardist2D.fromBioimageioModel(descriptor);
		} else if (descriptor.getModelFamily().equals(ModelDescriptor.BIOIMAGEIO)
				&& !descriptor.getWeights().getAllSuportedWeightNames().contains(ModelWeight.getPytorchID())) {
			model = BioimageIoModelJava.createBioimageioModel(descriptor.getModelPath(), enginesPath);
		} else if (descriptor.getModelFamily().equals(ModelDescriptor.BIOIMAGEIO)) {
			model = BioimageIoModelPytorch.create(descriptor);
		} else {
			throw new IllegalArgumentException("Model not supported");
		}
	}

	protected RunnerAdapter(ModelDescriptor descriptor, String enginesPath, ClassLoader classloader) 
			throws IOException, LoadEngineException {
		this.descriptor = descriptor;
		if (descriptor.getModelFamily().equals(ModelDescriptor.STARDIST)) {
			model = Stardist2D.fromBioimageioModel(descriptor);
		} else if (descriptor.getModelFamily().equals(ModelDescriptor.BIOIMAGEIO)
				&& !descriptor.getWeights().getAllSuportedWeightNames().contains(ModelWeight.getPytorchID())) {
			model = BioimageIoModelJava.createBioimageioModel(descriptor.getModelPath(), enginesPath, classloader);
		} else if (descriptor.getModelFamily().equals(ModelDescriptor.BIOIMAGEIO)) {
			model = BioimageIoModelPytorch.create(descriptor);
		} else {
			throw new IllegalArgumentException("Model not supported");
		}
	}
	
	/**
	 * 
	 * @return the model descriptor
	 */
	public ModelDescriptor getDescriptor() {
		return this.descriptor;
	}
	
	public void load() throws LoadModelException {
		if (closed)
			throw new RuntimeException("The model has already been closed");
		this.model.loadModel();
		this.loaded = true;
	}
	
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<R>> run(List<Tensor<T>> inputTensors) throws FileNotFoundException, RunModelException, IOException {
		if (closed)
			throw new RuntimeException("The model has already been closed");
		if (!this.model.isLoaded())
			throw new RuntimeException("Please first load the model");
		return model.run(inputTensors);
	}
	
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<R>> runOnTestImages() throws FileNotFoundException, ModelSpecsException, RunModelException, IOException {
		LinkedHashMap<TensorSpec, String> testInputs = getTestInputs();
		LinkedHashMap<TensorSpec, RandomAccessibleInterval<T>> inputRais = displayTestInputs(testInputs);
		List<Tensor<T>> inputTensors = createTestTensorList(inputRais);
		return model.run(inputTensors);
	}
	
	private  <T extends RealType<T> & NativeType<T>> 
	List<Tensor<T>> createTestTensorList(LinkedHashMap<TensorSpec, RandomAccessibleInterval<T>> inputRais){
		return inputRais.entrySet().stream()
				.map(ee -> Tensor.build(ee.getKey().getName(), ee.getKey().getAxesOrder(), ee.getValue()))
				.collect(Collectors.toList());
	}
	
	public boolean isClosed() {
		return this.closed;
	}
	
	public boolean isLoaded() {
		if (this.isClosed())
			return false;
		return this.loaded;
	}

	@Override
	public void close() throws IOException {
		model.close();
		closed = true;
		loaded = false;		
	}
}
