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
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import org.apache.commons.compress.archivers.ArchiveException;

import io.bioimage.modelrunner.apposed.appose.Environment;
import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.apposed.appose.Service;
import io.bioimage.modelrunner.apposed.appose.Service.Task;
import io.bioimage.modelrunner.apposed.appose.Service.TaskStatus;
import io.bioimage.modelrunner.bioimageio.tiling.TileInfo;
import io.bioimage.modelrunner.bioimageio.tiling.TileMaker;
import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.BaseModel;
import io.bioimage.modelrunner.model.java.DLModelJava.TilingConsumer;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;

public class DLModelPytorch extends BaseModel {
	
	protected final String modelFile;
	
	protected final String callable;
	
	protected final String weightsPath;
	
	protected final Map<String, Object> kwargs;
	
	public String envPath;
	
	private Service python;
	
	private List<SharedMemoryArray> inShmaList = new ArrayList<SharedMemoryArray>();
		
	private List<String> outShmNames;
	
	private List<String> outShmDTypes;
	
	private List<long[]> outShmDims;
	
	/**
	 * List containing the desired tiling strategy for each of the input tensors
	 */
	protected List<TileInfo> inputTiles;
	
	/**
	 * List containing the desired tiling strategy for each of the output tensors
	 */
	protected List<TileInfo> outputTiles;
	/**
	 * Whether to do tiling or not when doing inference
	 */
	protected boolean tiling = false;
	
	/**
	 * Consumer used to inform the current tile being processed and in how many
	 * tiles the input images are going to be separated
	 */
	protected TilingConsumer tileCounter;
	
	public static final String COMMON_PYTORCH_ENV_NAME = "biapy";
	
	private static final List<String> BIAPY_CONDA_DEPS = Arrays.asList(new String[] {"python=3.10"});
	
	private static final List<String> BIAPY_PIP_DEPS_TORCH = Arrays.asList(new String[] {"torch==2.4.0", 
			"torchvision==0.19.0", "torchaudio==2.4.0"});
	
	private static final List<String> BIAPY_PIP_DEPS = Arrays.asList(new String[] {"timm==1.0.14", "pytorch-msssim==1.0.0", 
			"torchmetrics[image]==1.4.*",
			"biapy==3.5.10", "appose"});
	
	private static final List<String> BIAPY_PIP_ARGS = Arrays.asList(new String[] {"--index-url", "https://download.pytorch.org/whl/cpu"});
		
	private static String INSTALLATION_DIR = Mamba.BASE_PATH;
	
	private static final String MODEL_VAR_NAME = "model_" + UUID.randomUUID().toString().replace("-", "_");

	protected static final String LOAD_MODEL_CODE_ABSTRACT = ""
			+ "if 'sys' not in globals().keys():" + System.lineSeparator()
			+ "  import sys" + System.lineSeparator()
			+ "  globals()['sys'] = sys" + System.lineSeparator()
			+ "if 'np' not in globals().keys():" + System.lineSeparator()
			+ "  import numpy as np" + System.lineSeparator()
			+ "  globals()['np'] = np" + System.lineSeparator()
			+ "if 'os' not in globals().keys():" + System.lineSeparator()
			+ "  import os" + System.lineSeparator()
			+ "  globals()['os'] = os" + System.lineSeparator()
			+ "if 'shared_memory' not in globals().keys():" + System.lineSeparator()
			+ "  from multiprocessing import shared_memory" + System.lineSeparator()
			+ "  globals()['shared_memory'] = shared_memory" + System.lineSeparator()
			+ "sys.path.append(os.path.abspath('%s'))" + System.lineSeparator()
			+ "from %s import %s" + System.lineSeparator();
	
	private static final String OUTPUT_LIST_KEY = "out_list" + UUID.randomUUID().toString().replace("-", "_");
	
	private static final String SHMS_KEY = "shms_" + UUID.randomUUID().toString().replace("-", "_");
	
	private static final String SHM_NAMES_KEY = "shm_names_" + UUID.randomUUID().toString().replace("-", "_");
	
	private static final String DTYPES_KEY = "dtypes_" + UUID.randomUUID().toString().replace("-", "_");
	
	private static final String DIMS_KEY = "dims_" + UUID.randomUUID().toString().replace("-", "_");
	
	protected static final String RECOVER_OUTPUTS_CODE = ""
			+ "def handle_output_list(out_list):" + System.lineSeparator()
			+ "  for i, outs_i in range(out_list):" + System.lineSeparator()
			+ "    if type(outs_i) == np.ndarray:" + System.lineSeparator()
			+ "      shm = shared_memory.SharedMemory(create=True, size=outs_i.nbytes)" + System.lineSeparator()
			+ "      sh_np_array = np.ndarray(outs_i.shape, dtype=outs_i.dtype, buffer=outs_i.buf)" + System.lineSeparator()
			+ "      np.copyto(sh_np_array, outs_i)" + System.lineSeparator()
			+ "      " + SHMS_KEY + ".append(shm)" + System.lineSeparator()
			+ "      " + SHM_NAMES_KEY + ".append(shm.name)" + System.lineSeparator()
			+ "      " + DTYPES_KEY + ".append(outs_i.dtype)" + System.lineSeparator()
			+ "      " + DIMS_KEY + ".append(outs_i.shape)" + System.lineSeparator()
			+ "    elif type(outs_i) == torch.tensor:" + System.lineSeparator()
			+ "      shm = shared_memory.SharedMemory(create=True, size=outs_i.numel() * outs_i.element_size())" + System.lineSeparator()
			+ "      np_arr = np.ndarray(outs_i.shape, dtype=outs_i.dtype.name, buffer=shm.buf)" + System.lineSeparator()
			+ "      tensor_np_view = torch.from_numpy(np_arr)" + System.lineSeparator()
			+ "      tensor_np_view.copy_(outs_i)" + System.lineSeparator()
			+ "      " + SHMS_KEY + ".append(shm)" + System.lineSeparator()
			+ "      " + SHM_NAMES_KEY + ".append(shm.name)" + System.lineSeparator()
			+ "      " + DTYPES_KEY + ".append(outs_i.dtype.name)" + System.lineSeparator()
			+ "      " + DIMS_KEY + ".append(outs_i.shape)" + System.lineSeparator()
			+ "    elif type(outs_i) == int:" + System.lineSeparator()
			+ "      shm = shared_memory.SharedMemory(create=True, size=8)" + System.lineSeparator()
			+ "      shm.buf[:8] = outs_i.to_bytes(8, byteorder='little', signed=True)" + System.lineSeparator()
			+ "      " + SHMS_KEY + ".append(shm)" + System.lineSeparator()
			+ "      " + SHM_NAMES_KEY + ".append(shm.name)" + System.lineSeparator()
			+ "      " + DTYPES_KEY + ".append('int64')" + System.lineSeparator()
			+ "      " + DIMS_KEY + ".append((1))" + System.lineSeparator()
			+ "    elif type(outs_i) == float:" + System.lineSeparator()
			+ "      shm = shared_memory.SharedMemory(create=True, size=8)" + System.lineSeparator()
			+ "      shm.buf[:8] = outs_i.to_bytes(8, byteorder='little', signed=True)" + System.lineSeparator()
			+ "      " + SHMS_KEY + ".append(shm)" + System.lineSeparator()
			+ "      " + SHM_NAMES_KEY + ".append(shm.name)" + System.lineSeparator()
			+ "      " + DTYPES_KEY + ".append('float64')" + System.lineSeparator()
			+ "      " + DIMS_KEY + ".append((1))" + System.lineSeparator()
			+ "    elif type(outs_i) == tuple or type(outs_i) == list:" + System.lineSeparator()
			+ "      handle_output_list(outs_i)" + System.lineSeparator()
			+ "    else:" + System.lineSeparator()
			+ "      task.update('output type : ' + str(type(outs_i)) + ' not supported. "
			+ "Only supported output types are: np.ndarray, torch.tensor, int and float, "
			+ "or a list or tuple of any of those.')" + System.lineSeparator()
			+ "" + System.lineSeparator()
			+ "" + System.lineSeparator()
			+ "globals()['handle_output_list'] = handle_output_list" + System.lineSeparator()
			+ "" + System.lineSeparator()
			+ "" + System.lineSeparator();

	private static final String CLEAN_SHM_CODE = ""
			+ "if '" + SHMS_KEY + "' in globals().keys():" + System.lineSeparator()
			+ "  for s in " + SHMS_KEY + ":" + System.lineSeparator()
			+ "    s.close()" + System.lineSeparator()
			+ "    s.unlink()" + System.lineSeparator()
			+ "    del s" + System.lineSeparator();
	
	protected DLModelPytorch(String modelFile, String callable, String weightsPath, 
			Map<String, Object> kwargs) throws IOException {
		if (new File(modelFile).isFile() == false || !modelFile.endsWith(".py"))
			throw new IllegalArgumentException("The model file does not correspond to an existing .py file.");
		if (new File(weightsPath).isFile() == false || !(weightsPath.endsWith(".pt") || weightsPath.endsWith(".pth")))
			throw new IllegalArgumentException("The weights file does not correspond to an existing .pt/.pth file.");
		this.callable = callable;
		this.modelFile = new File(modelFile).getAbsolutePath();
		this.weightsPath = new File(weightsPath).getAbsolutePath();
		this.kwargs = kwargs;
		this.envPath = INSTALLATION_DIR + File.separator + "envs" + File.separator + COMMON_PYTORCH_ENV_NAME;
		createPythonService();
	}
	
	public static DLModelPytorch create(String modelClass, String callable, 
			String modelPath) throws IOException {
		return create(modelClass, callable, modelPath, new HashMap<String, Object>());
	}
	
	public static DLModelPytorch create(String modelClass, String callable, 
			String modelPath, Map<String, Object> kwargs) throws IOException {
		return new DLModelPytorch(modelClass, callable, modelPath, kwargs);
	}
	
	protected void createPythonService() throws IOException {
		Environment env = new Environment() {
			@Override public String base() { return envPath; }
			};
		python = env.python();
		python.debug(System.err::println);
	}
	
	public void setCustomEnvPath(String envPath) throws IOException {
		this.envPath = envPath;
		this.python.close();
		createPythonService();
	}
	
	public boolean isTiling() {
		return this.tiling;
	}
	
	public void setTiling(boolean doTiling) {
		this.tiling = doTiling;
	}
	
	/**
	 * Set the wanted tile specifications for each of the input tensors.
	 * If this is not set, the model will process every input in just one run.
	 * however, if this is set, the model will always do tiling when running following
	 * this specifications
	 * 
	 * If this is called, automatically sets {@link #tiling} to true
	 * 
	 * @param inputTiles
	 * 	the specifications of how each of the input images can be tiled
	 * @param outputTiles
	 * 	the specifications of how each of the output images will be tiled
	 */
	public void setTileInfo(List<TileInfo> inputTiles, List<TileInfo> outputTiles) {
		this.inputTiles = inputTiles;
		this.outputTiles = outputTiles;
		this.tiling = true;
	}
	
	/**
	 * Set a consumer that can be used to get the number of tiles
	 * in which the input images will be separated and 
	 * the tile that is being processed
	 * @param tileCounter
	 * 	{@link TilingConsumer} used to track the inference of tiles
	 */
	public void setTilingCounter(TilingConsumer tileCounter) {
		this.tileCounter = tileCounter;
	}

	@Override
	public void loadModel() throws LoadModelException {
		if (loaded)
			return;
		String moduleName = new File(modelFile).getName();
		moduleName = moduleName.substring(0, moduleName.length() - 3);
		String code = String.format(LOAD_MODEL_CODE_ABSTRACT, new File(modelFile).getParentFile().getAbsolutePath(), moduleName, callable);
		code += buildModelCode();
		
		code += RECOVER_OUTPUTS_CODE;

		Task task;
		try {
			task = python.task(code);
			task.waitFor();
			if (task.status == TaskStatus.CANCELED)
				throw new RuntimeException("Task canceled");
			else if (task.status == TaskStatus.FAILED)
				throw new RuntimeException(task.error);
			else if (task.status == TaskStatus.CRASHED)
				throw new RuntimeException(task.error);
		} catch (IOException | InterruptedException e) {
			throw new LoadModelException(Types.stackTrace(e));
		}
		loaded = true;
	}
	
	private String buildModelCode() {
		String code = ""
				+ "if 'torch' not in globals().keys():" + System.lineSeparator()
				+ "  import torch" + System.lineSeparator()
				+ "  globals()['torch'] = torch" + System.lineSeparator();
		code = MODEL_VAR_NAME + "=" + callable + "(";
		for (Entry<String, Object> ee : kwargs.entrySet()) {
			code += ee.getKey() + "=" + ee.getValue() + ",";
		}
		code += ")" + System.lineSeparator();
		code += MODEL_VAR_NAME + ".load_state_dict("
				+ "torch.load('" + this.weightsPath + "', map_location=" + MODEL_VAR_NAME  + ".device)"
				+ ")" + System.lineSeparator();
		code += "globals()['" + MODEL_VAR_NAME + "'] = " + MODEL_VAR_NAME + System.lineSeparator();
		return code;
	}

	@Override
	public void close() {
		if (!loaded)
			return;
		python.close();
		loaded = false;
		
	}
	
	private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	Map<String, RandomAccessibleInterval<R>> predictForInputTensors(List<Tensor<T>> inTensors) 
	throws RunModelException {
		if (!loaded)
			throw new RuntimeException("Please load the model first.");
		return executeCode(createInputsCode(inTensors));		
	}
	
	private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	Map<String, RandomAccessibleInterval<R>> executeCode(String code) 
	throws RunModelException {
		Map<String, RandomAccessibleInterval<R>> outMap;
		try {
			Task task = python.task(code);
			task.waitFor();
			if (task.status == TaskStatus.CANCELED)
				throw new RuntimeException("Task canceled");
			else if (task.status == TaskStatus.FAILED)
				throw new RuntimeException(task.error);
			else if (task.status == TaskStatus.CRASHED)
				throw new RuntimeException(task.error);
			loaded = true;
			outMap = reconstructOutputs(task);
			cleanShm();
		} catch (IOException | InterruptedException e) {
			try {
				cleanShm();
			} catch (InterruptedException | IOException e1) {
				throw new RunModelException(Types.stackTrace(e1));
			}
			throw new RunModelException(Types.stackTrace(e));
		}
		return outMap;
	}
	
	private <T extends RealType<T> & NativeType<T>> String 
	createInpusCode(List<RandomAccessibleInterval<T>> rais) {
		String code = "";
		for (int i = 0; i < rais.size(); i ++) {
			SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(rais.get(i));
			code += codeToConvertShmaToPython(shma, "torch_rai_" + i);
			inShmaList.add(shma);
		}
		code += OUTPUT_LIST_KEY + " = " + MODEL_VAR_NAME + "(";
		for (int i = 0; i < rais.size(); i ++)
			code += "torch_rai_" + i;
		code = code.substring(0, code.length() - 2);
		code += ")" + System.lineSeparator();
		code += ""
				+ SHMS_KEY + " = []" + System.lineSeparator()
				+ SHM_NAMES_KEY + " = []" + System.lineSeparator()
				+ DTYPES_KEY + " = []" + System.lineSeparator()
				+ DIMS_KEY + " = []" + System.lineSeparator()
				+ "globals()['" + SHMS_KEY + "'] = " + SHMS_KEY + System.lineSeparator();
		code += "handle_output_list(OUTPUT_LIST_KEY)" + System.lineSeparator();
		return code;
	}
	
	/**
	 * Simply run inference on the images provided. If the dimensions, number, data type or other
	 * characteristic of the tensor is not correct, an exception will be thrown.
	 * @param <T>
	 * 	input data type
	 * @param <R>
	 * 	ouptut data type
	 * @param inputs
	 * 	the list of {@link RandomAccessibleInterval} that will be used as inputs
	 * @return a list of {@link RandomAccessibleInterval} that has been outputed by the model
	 * @throws RunModelException
	 *             if there is an error in the execution of the model
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<RandomAccessibleInterval<R>> inference(List<RandomAccessibleInterval<T>> inputs) throws RunModelException {

		if (!loaded)
			throw new RuntimeException("Please load the model first.");
		String code = createInpusCode(inputs);
		Map<String, RandomAccessibleInterval<R>> map = executeCode(code);
		List<RandomAccessibleInterval<R>> outRais = new ArrayList<RandomAccessibleInterval<R>>();
		for (Entry<String, RandomAccessibleInterval<R>> ee : map.entrySet()) {
			outRais.add(ee.getValue());
		}
		return outRais;
	}
	
	private <T extends RealType<T> & NativeType<T>> String createInputsCode(List<Tensor<T>> inTensors) {
		String code = "";
		for (Tensor<T> in : inTensors) {
			SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(in.getData());
			code += codeToConvertShmaToPython(shma, in.getName() + "_torch");
			inShmaList.add(shma);
		}
		code += OUTPUT_LIST_KEY + " = " + MODEL_VAR_NAME + "(";
		for (Tensor<T> in : inTensors)
			code += in.getName() + "_torch, ";
		code = code.substring(0, code.length() - 2);
		code += ")" + System.lineSeparator();
		code += ""
				+ SHMS_KEY + " = []" + System.lineSeparator()
				+ SHM_NAMES_KEY + " = []" + System.lineSeparator()
				+ DTYPES_KEY + " = []" + System.lineSeparator()
				+ DIMS_KEY + " = []" + System.lineSeparator()
				+ "globals()['" + SHMS_KEY + "'] = " + SHMS_KEY + System.lineSeparator();
		code += "handle_output_list(OUTPUT_LIST_KEY)" + System.lineSeparator();
		return code;
	}

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> run(List<Tensor<R>> inputTensors) throws RunModelException {
		if (!this.isLoaded())
			throw new RunModelException("Please first load the model.");
		if (!this.tiling) {
			throw new UnsupportedOperationException("Cannot run a DLModel if no information about the outputs is provided."
					+ " Either try with 'run( List< Tensor < T > > inTensors, List< Tensor < R > > outTensors )'"
					+ " or set the tiling information with 'setTileInfo(List<TileInfo> inputTiles, List<TileInfo> outputTiles)'."
					+ " Another option is to run simple inference over an ImgLib2 RandomAccessibleInterval with"
					+ " 'inference(List<RandomAccessibleInteral<T>> input)'");
		}
		if (this.isTiling() && (inputTiles != null || this.inputTiles.size() == 0))
			throw new UnsupportedOperationException("Tiling is set to 'true' but the input tiles are not well defined");
		else if (this.isTiling() && (this.outputTiles == null || this.outputTiles.size() == 0))
			throw new UnsupportedOperationException("Tiling is set to 'true' but the output tiles are not well defined");
		
		TileMaker maker = TileMaker.build(inputTiles, outputTiles);
		List<Tensor<T>> outTensors = createOutputTensors();
		runTiling(inputTensors, outTensors, maker);
		return outTensors;
	}
	
	private <T extends RealType<T> & NativeType<T>> List<Tensor<T>> createOutputTensors() {
		List<Tensor<T>> outputTensors = new ArrayList<Tensor<T>>();
		for (TileInfo tt : this.outputTiles) {
			outputTensors.add((Tensor<T>) Tensor.buildBlankTensor(tt.getName(), 
																	tt.getImageAxesOrder(), 
																	tt.getImageDims(), 
																	(T) new FloatType()));
		}
		return outputTensors;
	}

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	void run(List<Tensor<T>> inTensors, List<Tensor<R>> outTensors)
			throws RunModelException {
		if (!this.isLoaded())
			throw new RunModelException("Please first load the model.");
		if (!this.tiling) {
			this.runNoTiles(inTensors, outTensors);
			return;
		}
		if (this.isTiling() && (inputTiles != null || this.inputTiles.size() == 0))
			throw new UnsupportedOperationException("Tiling is set to 'true' but the input tiles are not well defined");
		else if (this.isTiling() && (this.outputTiles == null || this.outputTiles.size() == 0))
			throw new UnsupportedOperationException("Tiling is set to 'true' but the output tiles are not well defined");
		TileMaker tiles = TileMaker.build(inputTiles, outputTiles);
		for (int i = 0; i < tiles.getNumberOfTiles(); i ++) {
			Tensor<R> tt = outTensors.get(i);
			long[] expectedSize = tiles.getOutputImageSize(tt.getName());
			if (expectedSize == null) {
				throw new IllegalArgumentException("Tensor '" + tt.getName() + "' is missing in the outputs.");
			} else if (!tt.isEmpty() && Arrays.equals(expectedSize, tt.getData().dimensionsAsLongArray())) {
				throw new IllegalArgumentException("Tensor '" + tt.getName() + "' size is different than the expected size"
						+ " defined for the output image: " + Arrays.toString(tt.getData().dimensionsAsLongArray()) 
						+ " vs " + Arrays.toString(expectedSize) + ".");
			}
		}
		runTiling(inTensors, outTensors, tiles);
	}
	
	protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	void runTiling(List<Tensor<R>> inputTensors, List<Tensor<T>> outputTensors, TileMaker tiles) throws RunModelException {
		for (int i = 0; i < tiles.getNumberOfTiles(); i ++) {
			int nTile = 0 + i;
			List<Tensor<R>> inputTiles = inputTensors.stream()
					.map(tt -> tiles.getNthTileInput(tt, nTile)).collect(Collectors.toList());
			List<Tensor<T>> outputTiles = outputTensors.stream()
					.map(tt -> tiles.getNthTileOutput(tt, nTile)).collect(Collectors.toList());
			runNoTiles(inputTiles, outputTiles);
		}
	}

	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	void runNoTiles(List<Tensor<T>> inTensors, List<Tensor<R>> outTensors) throws RunModelException {
		Map<String, RandomAccessibleInterval<R>> outMap = predictForInputTensors(inTensors);
		int c = 0;
		for (Entry<String, RandomAccessibleInterval<R>> ee : outMap.entrySet()) {
			RandomAccessibleInterval<R> rai = ee.getValue();
			try {
				outTensors.get(c).setData(rai);
				c ++;
			} catch (Exception ex) {
			}
			
		}
	}
	
	private void closeShm() throws IOException {
		for (SharedMemoryArray shm : inShmaList) {
			shm.close();
		}
	}
	
	private void cleanShm() throws InterruptedException, IOException {
		closeShm();
		if (PlatformDetection.isWindows()) {
			Task closeSHMTask = python.task(CLEAN_SHM_CODE);
			closeSHMTask.waitFor();
		}
	}
	
	protected <T extends RealType<T> & NativeType<T>> 
	Map<String, RandomAccessibleInterval<T>> reconstructOutputs(Task task) throws IOException {
		buildOutShmList(task);
		buildOutDTypesList(task);
		buildOutDimsList(task);
		LinkedHashMap<String, RandomAccessibleInterval<T>> outs = new LinkedHashMap<String, RandomAccessibleInterval<T>>();
		for (int i = 0; i < this.outShmNames.size(); i ++) {
			String shmName = outShmNames.get(i);
			String dtype = outShmDTypes.get(i);
			long[] dims = outShmDims.get(i);
			RandomAccessibleInterval<T> rai = reconstruct(shmName, dtype, dims);
			outs.put("output_" + i, rai);
		}
		return outs;
	}
	
	private void buildOutShmList(Task task) {
		this.outShmNames = new ArrayList<String>();
		if (task.outputs.get(SHM_NAMES_KEY) instanceof List == false)
			throw new RuntimeException("Unexpected type for '" + SHM_NAMES_KEY + "'.");
		List<?> list = (List<?>) task.outputs.get(SHM_NAMES_KEY);
		for (Object elem : list) {
			if (elem instanceof String == false)
				throw new RuntimeException("Unexpected type for element of  '" + SHM_NAMES_KEY + "' list.");
			outShmNames.add((String) elem);
		}
	}
	
	private void buildOutDTypesList(Task task) {
		this.outShmDTypes = new ArrayList<String>();
		if (task.outputs.get(DTYPES_KEY) instanceof List == false)
			throw new RuntimeException("Unexpected type for '" + DTYPES_KEY + "'.");
		List<?> list = (List<?>) task.outputs.get(DTYPES_KEY);
		for (Object elem : list) {
			if (elem instanceof String == false)
				throw new RuntimeException("Unexpected type for element of  '" + DTYPES_KEY + "' list.");
			outShmDTypes.add((String) elem);
		}
	}
	
	private void buildOutDimsList(Task task) {
		this.outShmDims = new ArrayList<long[]>();
		if (task.outputs.get(DIMS_KEY) instanceof List == false)
			throw new RuntimeException("Unexpected type for '" + DIMS_KEY + "'.");
		List<?> list = (List<?>) task.outputs.get(DIMS_KEY);
		for (Object elem : list) {
			if (elem instanceof Object[] == false)
				throw new RuntimeException("Unexpected type for element of  '" + DIMS_KEY + "' list.");
			Object[] arr = (Object[]) elem;
			long[] longArr = new long[arr.length];
			for (int i = 0; i < arr.length; i ++) {
				if (arr[i] instanceof Number == false)
					throw new RuntimeException("Unexpected type for array of element of  '" + DIMS_KEY + "' list.");
				longArr[i] = ((Number) arr[i]).longValue();
			}
			outShmDims.add(longArr);
		}
	}
	
	private <T extends RealType<T> & NativeType<T>> 
	RandomAccessibleInterval<T> reconstruct(String key, String dtype, long[] dims) throws IOException {

		SharedMemoryArray shm = SharedMemoryArray.readOrCreate(key, dims, 
				Cast.unchecked(CommonUtils.getImgLib2DataType(dtype)), false, false);
		
		// TODO I do not understand why is complaining when the types align perfectly
		RandomAccessibleInterval<T> rai = shm.getSharedRAI();
		RandomAccessibleInterval<T> raiCopy = Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(rai), 
				Util.getTypeFromInterval(Cast.unchecked(rai)));
		
		shm.close();
		
		return raiCopy;
	}
	
	public static String codeToConvertShmaToPython(SharedMemoryArray shma, String varName) {
		String code = "";
		// This line wants to recreate the original numpy array. Should look like:
		// input0_appose_shm = shared_memory.SharedMemory(name=input0)
		// input0 = np.ndarray(size, dtype="float64", buffer=input0_appose_shm.buf).reshape([64, 64])
		code += varName + "_shm = shared_memory.SharedMemory(name='"
							+ shma.getNameForPython() + "', size=" + shma.getSize() 
							+ ")" + System.lineSeparator();
		long nElems = 1;
		for (long elem : shma.getOriginalShape()) nElems *= elem;
		code += varName + "_np = np.ndarray(" + nElems  + ", dtype='" + CommonUtils.getDataTypeFromRAI(Cast.unchecked(shma.getSharedRAI()))
			  + "', buffer=" + varName +"_shm.buf).reshape([";
		for (int i = 0; i < shma.getOriginalShape().length; i ++)
			code += shma.getOriginalShape()[i] + ", ";
		code += "])" + System.lineSeparator();
		code += varName + " = torch.from_numpy(" + varName + "_np).to(" + MODEL_VAR_NAME + ".device)" + System.lineSeparator();

		code += ""
			+ "if os.name == 'nt':" + System.lineSeparator()
			+ "  im_shm.close()" + System.lineSeparator()
			+ "  im_shm.unlink()" + System.lineSeparator();
		return code;
	}
	
	/**
	 * Check whether everything that is needed for Stardist 2D is installed or not
	 * @return true if the full python environment is installed or not
	 */
	public static boolean isInstalled() {
		// TODO
		return isInstalled(null);
	}
	
	/**
	 * Check whether everything that is needed for Stardist 2D is installed or not
	 * @return true if the full python environment is installed or not
	 */
	public static boolean isInstalled(String envPath) {
		// TODO
		return false;
	}
	
	/**
	 * Check whether the requirements needed to run a pytorch model are satisfied or not.
	 * First checks if the corresponding Java DL engine is installed or not, then checks
	 * if the Python environment needed for a pytorch model post processing is fine too.
	 * 
	 * If anything is not installed, this method also installs it
	 * 
	 * @throws IOException if there is any error downloading the DL engine or installing the micromamba environment
	 * @throws InterruptedException if the installation is stopped
	 * @throws RuntimeException if there is any unexpected error in the micromamba environment installation
	 * @throws MambaInstallException if there is any error downloading or installing micromamba
	 * @throws ArchiveException if there is any error decompressing the micromamba installer
	 * @throws URISyntaxException if the URL to the micromamba installation is not correct
	 */
	public static void installRequirements() throws IOException, InterruptedException, 
													RuntimeException, MambaInstallException, 
													ArchiveException, URISyntaxException {
		installRequirements(null);
	}
	
	/**
	 * Check whether the requirements needed to run a pytorch model are satisfied or not.
	 * First checks if the corresponding Java DL engine is installed or not, then checks
	 * if the Python environment needed for a pytorch model post processing is fine too.
	 * 
	 * If anything is not installed, this method also installs it
	 * 
	 * @param consumer
	 * 	String consumer that reads the installation log
	 * 
	 * @throws IOException if there is any error downloading the DL engine or installing the micromamba environment
	 * @throws InterruptedException if the installation is stopped
	 * @throws RuntimeException if there is any unexpected error in the micromamba environment installation
	 * @throws MambaInstallException if there is any error downloading or installing micromamba
	 * @throws ArchiveException if there is any error decompressing the micromamba installer
	 * @throws URISyntaxException if the URL to the micromamba installation is not correct
	 */
	public static void installRequirements(Consumer<String> consumer) throws IOException, InterruptedException, 
													RuntimeException, MambaInstallException, 
													ArchiveException, URISyntaxException {
		
		Mamba mamba = new Mamba(INSTALLATION_DIR);
		if (consumer != null) {
			mamba.setConsoleOutputConsumer(consumer);
			mamba.setErrorOutputConsumer(consumer);
		}
		boolean biapyPythonInstalled = false;
		try {
			biapyPythonInstalled = mamba.checkAllDependenciesInEnv(COMMON_PYTORCH_ENV_NAME, BIAPY_CONDA_DEPS);
			biapyPythonInstalled = mamba.checkAllDependenciesInEnv(COMMON_PYTORCH_ENV_NAME, BIAPY_PIP_DEPS);
		} catch (MambaInstallException e) {
			mamba.installMicromamba();
		}
		if (!biapyPythonInstalled) {
			// TODO add logging for environment installation
			mamba.create(COMMON_PYTORCH_ENV_NAME, true, new ArrayList<String>(), BIAPY_CONDA_DEPS);
			ArrayList<String> args = new ArrayList<String>(BIAPY_PIP_ARGS);
			args.addAll(BIAPY_PIP_DEPS_TORCH);
			mamba.pipInstallIn(COMMON_PYTORCH_ENV_NAME, args.toArray(new String[args.size()]));
			mamba.pipInstallIn(COMMON_PYTORCH_ENV_NAME, BIAPY_PIP_DEPS.toArray(new String[BIAPY_PIP_DEPS.size()]));
		};
	}
	
	/**
	 * Set the directory where the Python for Pytorch environment will be installed
	 * @param installationDir
	 * 	directory where the Python for Pytorch environment will be created
	 */
	public static void setInstallationDir(String installationDir) {
		INSTALLATION_DIR = installationDir;
	}
	
	/**
	 * 
	 * @return the directory where the Python for Pytorch environment will be created
	 */
	public static String getInstallationDir() {
		return INSTALLATION_DIR;
	}

}
