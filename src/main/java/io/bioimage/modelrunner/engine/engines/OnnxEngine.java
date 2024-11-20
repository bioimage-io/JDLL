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
package io.bioimage.modelrunner.engine.engines;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.compress.archivers.ArchiveException;

import io.bioimage.modelrunner.apposed.appose.Environment;
import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.apposed.appose.Service;
import io.bioimage.modelrunner.apposed.appose.Service.Task;
import io.bioimage.modelrunner.apposed.appose.Service.TaskStatus;
import io.bioimage.modelrunner.engine.AbstractEngine;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class OnnxEngine extends AbstractEngine {
	
	private Mamba mamba;
	
	private String version;
	
	private boolean gpu;
	
	private boolean isPython;
	
	private Boolean installed;

	private Environment env;
	
	private Service python;
	
	public static final String NAME = "onnx";

	private static final List<String> SUPPORTED_ONNX_GPU_VERSIONS = Arrays.stream(new String[] {}).collect(Collectors.toList());
	private static final List<String> SUPPORTED_ONNX_VERSION_NUMBERS = Arrays.stream(new String[] {}).collect(Collectors.toList());
	
	private static final Map<String, String> LOAD_SCRIPT_MAP;
	
	static {
		LOAD_SCRIPT_MAP = new HashMap<String, String>();
		LOAD_SCRIPT_MAP.put("", "");
		LOAD_SCRIPT_MAP.put("", "");
	}

	private static final String UNLOAD_SCRIPT_KERAS_2 = "";
	
	private static final String UNLOAD_SCRIPT_KERAS_3 = "";
	
	private static final Map<String, String> UNLOAD_SCRIPT_MAP;
	
	static {
		UNLOAD_SCRIPT_MAP = new HashMap<String, String>();
		UNLOAD_SCRIPT_MAP.put("", UNLOAD_SCRIPT_KERAS_2);
		UNLOAD_SCRIPT_MAP.put("", UNLOAD_SCRIPT_KERAS_3);
	}
	
	private static final String IS_MODEL_LOADED_SCRIPT_KERAS_2 = "";
	
	private static final String IS_MODEL_LOADED_SCRIPT_KERAS_3 = "";
	
	private static final Map<String, String> IS_MODEL_LOADED_SCRIPT_MAP;
	
	static {
		IS_MODEL_LOADED_SCRIPT_MAP = new HashMap<String, String>();
		IS_MODEL_LOADED_SCRIPT_MAP.put("", IS_MODEL_LOADED_SCRIPT_KERAS_2);
		IS_MODEL_LOADED_SCRIPT_MAP.put("", IS_MODEL_LOADED_SCRIPT_KERAS_3);
	}
	
	private static final String RUN_SCRIPT_KERAS_2 = "";
	
	private static final String RUN_SCRIPT_KERAS_3 = "";
	
	private static final Map<String, String> RUN_SCRIPT_MAP;
	
	static {
		RUN_SCRIPT_MAP = new HashMap<String, String>();
		RUN_SCRIPT_MAP.put("", RUN_SCRIPT_KERAS_2);
		RUN_SCRIPT_MAP.put("", RUN_SCRIPT_KERAS_3);
	}
	
	private OnnxEngine(String version, boolean gpu, boolean isPython) {
		if (!isPython) 
			throw new IllegalArgumentException("JDLL only has support for Onnx through a Java engine.");
		if (!SUPPORTED_ONNX_VERSION_NUMBERS.contains(version))
			throw new IllegalArgumentException("The provided Onnx version is not supported by JDLL: " + version
					+ ". The supported versions are: " + SUPPORTED_ONNX_VERSION_NUMBERS);
		if (gpu && !SUPPORTED_ONNX_GPU_VERSIONS.contains(version))
			throw new IllegalArgumentException("The provided Onnx version has no GPU support in JDLL: " + version
					+ ". GPU supported versions are: " + SUPPORTED_ONNX_GPU_VERSIONS);
		mamba = new Mamba();
		this.isPython = isPython;
		this.version = version;
	}

	
	public static OnnxEngine initialize(String version, boolean gpu, boolean isPython) {
		return new OnnxEngine(version, gpu, isPython);
	}
	
	public static List<OnnxEngine> getInstalledVersions() {
		return null;
	}
	
	@Override
	public String getName() {
		return NAME;
	}
	
	@Override
	public String getDir() {
		return mamba.getEnvsDir() + File.separator + this.toString();
	}


	@Override
	public boolean isPython() {
		return isPython;
	}


	@Override
	public String getVersion() {
		return version;
	}


	@Override
	public boolean supportsGPU() {
		return gpu;
	}


	@Override
	public boolean isInstalled() {
		if (installed != null)
			return installed;
		if (!(new File(getDir()).exists()))
			return false;
		installed = getInstalledVersions().stream()
				.filter(vv -> vv.gpu == gpu && vv.version.equals(version)).findFirst().orElse(null) != null;
		return installed;
	}


	@Override
	public void install() throws IOException, InterruptedException, MambaInstallException, ArchiveException, URISyntaxException {
		if (!mamba.checkMambaInstalled()) mamba.installMicromamba();
		
		mamba.create(getDir(), getSupportedEngineKeys());
		installed = true;
	}

	@Override
	public void loadModel(String modelFolder, String modelSource) throws IOException, InterruptedException {
		if (!this.isInstalled())
			throw new IllegalArgumentException("Current engine '" + this.toString() 
												+ "' is not installed. Please install it first.");
		if (env == null) {
			this.env = new Environment() {
				@Override public String base() { return OnnxEngine.this.getDir(); }
				@Override public boolean useSystemPath() { return false; }
				};
			python = env.python();
		}
		String loadScriptFormatted = String.format(LOAD_SCRIPT_MAP.get(this.version), modelFolder, modelSource);
		Task task = python.task(loadScriptFormatted);
		task.waitFor();
		if (task.status == TaskStatus.COMPLETE)
			return;
		throw new RuntimeException("Error loading the model. " + task.error);
	}

	@Override
	public boolean isModelLoaded(String modelFolder, String modelSource) throws IOException, InterruptedException {
		if (python == null)
			return false;
		String loadScriptFormatted = String.format(IS_MODEL_LOADED_SCRIPT_MAP.get(this.version));
		Task task = python.task(loadScriptFormatted);
		task.waitFor();
		if (task.status == TaskStatus.COMPLETE)
			return task.outputs.get("isLoaded").equals("True");
		throw new RuntimeException("Error unloading the model. " + task.error);	
	}

	@Override
	public <T extends RealType<T> & NativeType<T>> void runModel(List<Tensor<T>> inputTensors, List<Tensor<T>> outputTensors)
			throws IOException, InterruptedException {
		if (python == null)
			throw new RuntimeException("Python Keras engine has not been loaded yet.");
		List<SharedMemoryArray> inputShms = inputTensors.stream()
				.map(tt -> SharedMemoryArray.createSHMAFromRAI(tt.getData(), false, false)).collect(Collectors.toList());
		List<Object> outputShms = inputTensors.stream()
				.map(tt -> {
					if (tt.isEmpty() && PlatformDetection.isWindows())
						return SharedMemoryArray.create(0);
					else if (tt.isEmpty())
						return SharedMemoryArray.createShmName();
					else
						return SharedMemoryArray.createSHMAFromRAI(tt.getData(), false, false);
				}).collect(Collectors.toList());
		String runScriptFormatted = createScriptForInference(inputShms, outputShms);
		Task task = python.task(runScriptFormatted);
		task.waitFor();
		if (task.status != TaskStatus.COMPLETE)
			throw new RuntimeException("Error making inference with the model. " + task.error);
		retrieveOutputs(outputShms, outputTensors);
	}

	private String createScriptForInference(List<SharedMemoryArray> inputs, List<Object> outputs) {
		String runScriptFormatted = String.format(RUN_SCRIPT_MAP.get(this.version));
		return "";
	}

	private <T extends RealType<T> & NativeType<T>> 
	void retrieveOutputs(List<Object> outputShms, List<Tensor<T>> outputTensors) {
		String retrieveOutputsScriptFormatted = String.format(RUN_SCRIPT_MAP.get(this.version));
	}
	
	@Override
	public void unloadModel() throws IOException, InterruptedException {
		if (python == null)
			return;
		String loadScriptFormatted = String.format(UNLOAD_SCRIPT_MAP.get(this.version));
		Task task = python.task(loadScriptFormatted);
		task.waitFor();
		if (task.status == TaskStatus.COMPLETE)
			return;
		throw new RuntimeException("Error unloading the model. " + task.error);		
	}

	@Override
	public void close() throws Exception {
		if (this.env == null && this.python == null)
			return;
		this.unloadModel();
		this.python.close();
		this.python = null;
		this.env = null;
		
	}
	
	@Override
	public String toString() {
		return NAME + "_" + version + (gpu ? "_gpu" : "");
	}

}
