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
import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.apposed.appose.TaskException;

import io.bioimage.modelrunner.engine.AbstractEngine;
import io.bioimage.modelrunner.model.python.DLModelPytorch;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class PytorchEngine extends AbstractEngine {
	
	
	private String version;
	
	private boolean gpu;
	
	private boolean isPython;
	
	private Boolean installed;

	private Environment env;
	
	private Service python;
	
	private String envString;
	
	public static final String NAME = "pytorch";

	private static final List<String> SUPPORTED_PYTORCH_GPU_VERSIONS = Arrays.stream(new String[] {}).collect(Collectors.toList());
	private static final List<String> SUPPORTED_PYTORCH_VERSION_NUMBERS = Arrays.stream(new String[] {}).collect(Collectors.toList());
	
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
	
	private PytorchEngine(String version, boolean gpu, boolean isPython) {
		if (!SUPPORTED_PYTORCH_VERSION_NUMBERS.contains(version))
			throw new IllegalArgumentException("The provided Pytorch version is not supported by JDLL: " + version
					+ ". The supported versions are: " + SUPPORTED_PYTORCH_VERSION_NUMBERS);
		if (gpu && !SUPPORTED_PYTORCH_GPU_VERSIONS.contains(version))
			throw new IllegalArgumentException("The provided Pytorch version has no GPU support in JDLL: " + version
					+ ". GPU supported versions are: " + SUPPORTED_PYTORCH_GPU_VERSIONS);
		this.isPython = isPython;
		this.version = version;
		if (false)
			this.envString = "cuda";
		else
			this.envString = DLModelPytorch.COMMON_PYTORCH_ENV_NAME;
	}

	
	/**
	 * Initializes ialize.
	 *
	 * @param version the version parameter.
	 * @param gpu the gpu parameter.
	 * @param isPython the isPython parameter.
	 * @return the resulting value.
	 */
	public static PytorchEngine initialize(String version, boolean gpu, boolean isPython) {
		return new PytorchEngine(version, gpu, isPython);
	}
	
	/**
	 * Gets installed versions.
	 *
	 * @return the resulting list.
	 */
	public static List<PytorchEngine> getInstalledVersions() {
		return null;
	}
	
	/**
	 * Gets name.
	 *
	 * @return the resulting string.
	 */
	@Override
	public String getName() {
		return NAME;
	}
	
	/**
	 * Gets dir.
	 *
	 * @return the resulting string.
	 */
	@Override
	public String getDir() {
		return new File(DLModelPytorch.getInstallationDir(), DLModelPytorch.COMMON_PYTORCH_ENV_NAME).toString();
	}


	/**
	 * Checks whether python.
	 *
	 * @return true if the operation succeeds; otherwise, false.
	 */
	@Override
	public boolean isPython() {
		return isPython;
	}


	/**
	 * Gets version.
	 *
	 * @return the resulting string.
	 */
	@Override
	public String getVersion() {
		return version;
	}


	/**
	 * Executes supports gpu.
	 *
	 * @return true if the operation succeeds; otherwise, false.
	 */
	@Override
	public boolean supportsGPU() {
		return gpu;
	}


	/**
	 * Checks whether installed.
	 *
	 * @return true if the operation succeeds; otherwise, false.
	 */
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


	/**
	 * Executes install.
	 *
	 * @throws IOException if an I/O error occurs.
	 * @throws InterruptedException if the current thread is interrupted while waiting for the operation to finish.
	 * @throws ArchiveException if a ArchiveException occurs while executing this method.
	 * @throws URISyntaxException if a URISyntaxException occurs while executing this method.
	 */
	@Override
	public void install() {
		installed = true;
	}

	/**
	 * Loads model.
	 *
	 * @param modelFolder the modelFolder parameter.
	 * @param modelSource the modelSource parameter.
	 * @throws IOException if an I/O error occurs.
	 * @throws InterruptedException if the current thread is interrupted while waiting for the operation to finish.
	 * @throws BuildException 
	 */
	@Override
	public void loadModel(String modelFolder, String modelSource) throws InterruptedException, BuildException, TaskException {
		if (!this.isInstalled())
			throw new IllegalArgumentException("Current engine '" + this.toString() 
												+ "' is not installed. Please install it first.");
		if (env == null) {
			this.env = Appose.pixi().environment(envString).wrap(new File(getDir()));
			python = env.python();
		}
		String loadScriptFormatted = String.format(LOAD_SCRIPT_MAP.get(this.version), modelFolder, modelSource);
		Task task = python.task(loadScriptFormatted);
		task.waitFor();
		if (task.status == TaskStatus.COMPLETE)
			return;
		throw new RuntimeException("Error loading the model. " + task.error);
	}

	/**
	 * Checks whether model loaded.
	 *
	 * @param modelFolder the modelFolder parameter.
	 * @param modelSource the modelSource parameter.
	 * @return true if the operation succeeds; otherwise, false.
	 * @throws InterruptedException if the current thread is interrupted while waiting for the operation to finish.
	 * @throws TaskException if there is any error running a task
	 */
	@Override
	public boolean isModelLoaded(String modelFolder, String modelSource) throws InterruptedException, TaskException {
		if (python == null)
			return false;
		String loadScriptFormatted = String.format(IS_MODEL_LOADED_SCRIPT_MAP.get(this.version));
		Task task = python.task(loadScriptFormatted);
		task.waitFor();
		if (task.status == TaskStatus.COMPLETE)
			return task.outputs.get("isLoaded").equals("True");
		throw new RuntimeException("Error unloading the model. " + task.error);	
	}

	/**
	 * Runs model.
	 *
	 * @param inputTensors the inputTensors parameter.
	 * @param outputTensors the outputTensors parameter.
	 * @throws TaskException if thre is an error running the model in python
	 * @throws InterruptedException if the current thread is interrupted while waiting for the operation to finish.
	 */
	@Override
	public <T extends RealType<T> & NativeType<T>> void runModel(List<Tensor<T>> inputTensors, List<Tensor<T>> outputTensors)
			throws TaskException, InterruptedException {
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
	
	/**
	 * Executes unload model.
	 *
	 * @throws TaskException if there is an error unloading the model
	 * @throws InterruptedException if the current thread is interrupted while waiting for the operation to finish.
	 */
	@Override
	public void unloadModel() throws TaskException, InterruptedException {
		if (python == null)
			return;
		String loadScriptFormatted = String.format(UNLOAD_SCRIPT_MAP.get(this.version));
		Task task = python.task(loadScriptFormatted);
		task.waitFor();
		if (task.status == TaskStatus.COMPLETE)
			return;
		throw new RuntimeException("Error unloading the model. " + task.error);		
	}

	/**
	 * Executes close.
	 *
	 * @throws Exception if the operation fails.
	 */
	@Override
	public void close() throws Exception {
		if (this.env == null && this.python == null)
			return;
		this.unloadModel();
		this.python.close();
		this.python = null;
		this.env = null;
		
	}
	
	/**
	 * Executes to string.
	 *
	 * @return the resulting string.
	 */
	@Override
	public String toString() {
		return NAME + "_" + version + (gpu ? "_gpu" : "");
	}

}
