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
package io.bioimage.modelrunner.gui.adapter;

import java.io.Closeable;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import org.apache.commons.compress.archivers.ArchiveException;

import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.bioimageio.description.exceptions.ModelSpecsException;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.engine.installation.EngineInstall;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.BaseModel;
import io.bioimage.modelrunner.model.java.BioimageIoModelJava;
import io.bioimage.modelrunner.model.python.BioimageIoModelPytorch;
import io.bioimage.modelrunner.model.python.DLModelPytorchProtected;
import io.bioimage.modelrunner.model.special.stardist.Stardist2D;
import io.bioimage.modelrunner.model.special.stardist.StardistAbstract;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.versionmanagement.InstalledEngines;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public abstract class RunnerAdapter implements Closeable {

	protected final ModelDescriptor descriptor;
	
	protected final String enginesPath;
	
	protected final ClassLoader classLoader;
	
	protected BaseModel model;
		
	protected boolean closed = false;
	
	protected boolean loaded = false;

	
	protected abstract <T extends RealType<T> & NativeType<T>>
	LinkedHashMap<TensorSpec, RandomAccessibleInterval<T>> displayTestInputs(LinkedHashMap<TensorSpec, String> testInputs);
	
	protected abstract LinkedHashMap<TensorSpec, String> getTestInputs();

	protected RunnerAdapter(ModelDescriptor descriptor) {
		this.descriptor = descriptor;
		this.enginesPath = new File("engines").getAbsolutePath();
		this.classLoader = null;
	}

	protected RunnerAdapter(ModelDescriptor descriptor, ClassLoader classloader) {
		this.descriptor = descriptor;
		this.enginesPath = new File("engines").getAbsolutePath();
		this.classLoader = classloader;
	}

	protected RunnerAdapter(ModelDescriptor descriptor, String enginesPath) {
		this.descriptor = descriptor;
		this.enginesPath = enginesPath;
		this.classLoader = null;
	}

	protected RunnerAdapter(ModelDescriptor descriptor, String enginesPath, ClassLoader classloader) {
		this.descriptor = descriptor;
		this.enginesPath = enginesPath;
		this.classLoader = classloader;
	}
	
	/**
	 * 
	 * @return the model descriptor
	 */
	public ModelDescriptor getDescriptor() {
		return this.descriptor;
	}
	
	public void load() throws LoadModelException {
		load(true);
	}
	
	public void load(boolean installIfMissing) throws LoadModelException {
		if (closed)
			throw new RuntimeException("The model has already been closed");
		try {
			if (this.classLoader == null)
				this.initWithEnginesPath(installIfMissing);
			else
				this.initWithEnginesClassLoader(installIfMissing);
		} catch (Exception ex) {
			throw new LoadModelException(Types.stackTrace(ex));
		}
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
	
	private void initWithEnginesPath(boolean install) throws IOException, LoadEngineException, InterruptedException, RuntimeException, MambaInstallException, ArchiveException, URISyntaxException {
		List<String> wList = descriptor.getWeights().getAllSuportedWeightNames();
		if (descriptor.getModelFamily().equals(ModelDescriptor.STARDIST)) {
			if (install && !StardistAbstract.isInstalled())
				StardistAbstract.installRequirements();
			model = Stardist2D.fromBioimageioModel(descriptor);
		} else if (descriptor.getModelFamily().equals(ModelDescriptor.BIOIMAGEIO)
				&& !(wList.size() == 1 && wList.contains(ModelWeight.getPytorchID()))) {
			if (install) {
				Consumer<Double> cons = (dd) -> {
					double progress = Math.round(dd * 10000) / 100;
					System.out.println("Downloading engines for " + descriptor.getName() + ":" + progress + "%");
				};
				EngineInstall.installEnginesForModelInDir(descriptor, enginesPath, cons);
			}
			model = BioimageIoModelJava.createBioimageioModel(descriptor.getModelPath(), enginesPath);
		} else if (descriptor.getModelFamily().equals(ModelDescriptor.BIOIMAGEIO)) {
			if (install && !DLModelPytorchProtected.isInstalled())
				DLModelPytorchProtected.installRequirements();
			model = BioimageIoModelPytorch.create(descriptor);
		} else {
			throw new IllegalArgumentException("Model not supported");
		}
	}
	
	private void initWithEnginesClassLoader(boolean install) throws LoadEngineException, IOException, InterruptedException, RuntimeException, MambaInstallException, ArchiveException, URISyntaxException {
		List<String> wList = descriptor.getWeights().getAllSuportedWeightNames();
		if (descriptor.getModelFamily().equals(ModelDescriptor.STARDIST)) {
			if (install && !StardistAbstract.isInstalled())
				StardistAbstract.installRequirements();
			model = Stardist2D.fromBioimageioModel(descriptor);
		} else if (descriptor.getModelFamily().equals(ModelDescriptor.BIOIMAGEIO)
				&& !(wList.size() == 1 && wList.contains(ModelWeight.getPytorchID()))) {
			if (install) {
				Consumer<Double> cons = (dd) -> {
					double progress = Math.round(dd * 10000) / 100;
					System.out.println("Downloading engines for " + descriptor.getName() + ":" + progress + "%");
				};
				EngineInstall.installEnginesForModelInDir(descriptor, enginesPath, cons);
			}
			model = BioimageIoModelJava.createBioimageioModel(descriptor.getModelPath(), enginesPath, this.classLoader);
		} else if (descriptor.getModelFamily().equals(ModelDescriptor.BIOIMAGEIO)) {
			if (install && !DLModelPytorchProtected.isInstalled())
				DLModelPytorchProtected.installRequirements();
			model = BioimageIoModelPytorch.create(descriptor);
		} else {
			throw new IllegalArgumentException("Model not supported");
		}
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
