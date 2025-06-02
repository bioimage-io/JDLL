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
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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

public class DLModelPytorchProtected extends BaseModel {
	
	protected final String modelFile;
	
	protected final String callable;
	
	protected final String importModule;
	
	protected final String weightsPath;
	
	protected final Map<String, Object> kwargs;
	
	protected String envPath;
	
	private Service python;
	
	protected List<SharedMemoryArray> inShmaList = new ArrayList<SharedMemoryArray>();
		
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
	
	protected static final boolean IS_ARM = PlatformDetection.isMacOS() 
			&& (PlatformDetection.getArch().equals(PlatformDetection.ARCH_ARM64) || PlatformDetection.isUsingRosseta());
	
	private static final List<String> BIAPY_CONDA_DEPS = Arrays.asList(new String[] {"python=3.10"});
	
	private static final List<String> BIAPY_PIP_DEPS_TORCH;
	static {
		if (PlatformDetection.isMacOS()
				&& PlatformDetection.getArch().equals(PlatformDetection.ARCH_X86_64) && !PlatformDetection.isUsingRosseta())
			BIAPY_PIP_DEPS_TORCH = Arrays.asList(new String[] {"torch==2.2.2", 
					"torchvision==0.17.2", "torchaudio==2.2.2"});
		else if (PlatformDetection.isWindows())
			BIAPY_PIP_DEPS_TORCH = Arrays.asList(new String[] {"torch==2.4.1", 
					"torchvision==0.19.1", "torchaudio==2.4.1"});
		else
			BIAPY_PIP_DEPS_TORCH = Arrays.asList(new String[] {"torch==2.4.0", 
					"torchvision==0.19.0", "torchaudio==2.4.0"});
	}
	
	private static final List<String> BIAPY_PIP_DEPS;
	static {
		if (PlatformDetection.isWindows())
			BIAPY_PIP_DEPS = Arrays.asList(new String[] {"timm==1.0.14",
					"pytorch-msssim==1.0.0", "torchmetrics==1.4.3", "cellpose==3.1.1.1", "scipy==1.15.2", "torch-fidelity==0.3.0",
					"careamics", "biapy==3.5.10", "appose"});
		else if (PlatformDetection.isMacOS() && PlatformDetection.getOSVersion().getMajor() < 14)
			BIAPY_PIP_DEPS = Arrays.asList(new String[] {"timm==1.0.14",
					"pytorch-msssim==1.0.0", "torchmetrics==1.4.3", "cellpose==3.1.1.1", "torch-fidelity==0.3.0",
					"careamics", "pooch>=1.8.1", "numpy<2", "imagecodecs>=2024.1.1", "bioimageio.core==0.7.0",
					"h5py>=3.9.0","torchinfo>=1.8.0", "pandas>=1.5.3", "xarray==2025.1.2",
					"fill-voids>=2.0.6", "edt>=2.3.2", "tqdm>=4.66.1", "yacs>=0.1.8", "zarr>=2.16.1",
					"pydot>=1.4.2", "matplotlib>=3.7.1", "imgaug>=0.4.0", "scipy==1.15.2",
					"tensorboardX>=2.6.2.2", "scikit-learn>=1.4.0", "opencv-python>=4.8.0.76", "scikit-image>=0.21.0",
					"appose"});
		else
			BIAPY_PIP_DEPS = Arrays.asList(new String[] {"timm==1.0.14",
					"pytorch-msssim==1.0.0", "torchmetrics==1.4.3", "cellpose==3.1.1.1", "scipy==1.15.2", "torch-fidelity==0.3.0",
					"careamics", "biapy==3.5.10", "appose"});
	}
	
	private static final List<String> BIAPY_PIP_ARGS = Arrays.asList(new String[] {"--index-url", "https://download.pytorch.org/whl/cpu"});
		
	protected static String INSTALLATION_DIR = Mamba.BASE_PATH;
	
	protected static final String MODEL_VAR_NAME = "model_" + UUID.randomUUID().toString().replace("-", "_");

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
			+ "%s" + System.lineSeparator()
			+ "%s" + System.lineSeparator()
			+ "if '%s' not in globals().keys():" + System.lineSeparator()
			+ "  globals()['%s'] = %s" + System.lineSeparator();
	
	protected static final String OUTPUT_LIST_KEY = "out_list" + UUID.randomUUID().toString().replace("-", "_");
	
	protected static final String SHMS_KEY = "shms_" + UUID.randomUUID().toString().replace("-", "_");
	
	protected static final String SHM_NAMES_KEY = "shm_names_" + UUID.randomUUID().toString().replace("-", "_");
	
	protected static final String DTYPES_KEY = "dtypes_" + UUID.randomUUID().toString().replace("-", "_");
	
	protected static final String DIMS_KEY = "dims_" + UUID.randomUUID().toString().replace("-", "_");
		
	protected static final String RECOVER_OUTPUTS_CODE = ""
			+ "def handle_output(outs_i):" + System.lineSeparator()
			+ "    if type(outs_i) == np.ndarray:" + System.lineSeparator()
			+ "      shm = shared_memory.SharedMemory(create=True, size=outs_i.nbytes)" + System.lineSeparator()
			+ "      sh_np_array = np.ndarray(outs_i.shape, dtype=outs_i.dtype, buffer=shm.buf)" + System.lineSeparator()
			+ "      np.copyto(sh_np_array, outs_i)" + System.lineSeparator()
			+ "      " + SHMS_KEY + ".append(shm)" + System.lineSeparator()
			+ "      " + SHM_NAMES_KEY + ".append(shm.name)" + System.lineSeparator()
			+ "      " + DTYPES_KEY + ".append(str(outs_i.dtype))" + System.lineSeparator()
			+ "      " + DIMS_KEY + ".append(outs_i.shape)" + System.lineSeparator()
			+ "    elif str(type(outs_i)) == \"<class 'torch.Tensor'>\":" + System.lineSeparator()
			+ "      if 'torch' not in globals().keys():" + System.lineSeparator()
			+ "        import torch" + System.lineSeparator()
			+ "        globals()['torch'] = torch" + System.lineSeparator()
			+ (!IS_ARM ? "" 
					: "        if torch.backends.mps.is_built() and torch.backends.mps.is_available():" + System.lineSeparator()
					+ "          device = 'mps'" + System.lineSeparator())
			+ "      else:" + System.lineSeparator()
			+ "        torch = globals()['torch']" + System.lineSeparator()
			+ "      shm = shared_memory.SharedMemory(create=True, size=outs_i.numel() * outs_i.element_size())" + System.lineSeparator()
			+ "      np_arr = np.ndarray(outs_i.shape, dtype=str(outs_i.dtype).split('.')[-1], buffer=shm.buf)" + System.lineSeparator()
			+ "      tensor_np_view = torch.from_numpy(np_arr)" + System.lineSeparator()
			+ "      tensor_np_view.copy_(outs_i)" + System.lineSeparator()
			+ "      " + SHMS_KEY + ".append(shm)" + System.lineSeparator()
			+ "      " + SHM_NAMES_KEY + ".append(shm.name)" + System.lineSeparator()
			+ "      " + DTYPES_KEY + ".append(str(outs_i.dtype).split('.')[-1])" + System.lineSeparator()
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
			+ System.lineSeparator()
			+ System.lineSeparator()
			+ "def handle_output_list(out_list):" + System.lineSeparator()
			+ "  if type(out_list) == tuple or type(out_list) == list:" + System.lineSeparator()
			+ "    for outs_i in out_list:" + System.lineSeparator()
			+ "      handle_output(outs_i)" + System.lineSeparator()
			+ "  else:" + System.lineSeparator()
			+ "    handle_output(out_list)" + System.lineSeparator()
			+ "" + System.lineSeparator()
			+ "" + System.lineSeparator()
			+ "globals()['handle_output_list'] = handle_output_list" + System.lineSeparator()
			+ "globals()['handle_output'] = handle_output" + System.lineSeparator()
			+ "" + System.lineSeparator()
			+ "" + System.lineSeparator();

	private static final String CLEAN_SHM_CODE = ""
			+ "if '" + SHMS_KEY + "' in globals().keys():" + System.lineSeparator()
			+ "  for s in " + SHMS_KEY + ":" + System.lineSeparator()
			+ "    s.close()" + System.lineSeparator()
			+ "    s.unlink()" + System.lineSeparator()
			+ "    del s" + System.lineSeparator();
	
	private static final String JDLL_UUID = UUID.randomUUID().toString().replaceAll("-", "_");
	
	protected DLModelPytorchProtected(String modelFile, String callable, String importModule, String weightsPath, 
			Map<String, Object> kwargs) throws IOException {
		this(modelFile, callable, importModule, weightsPath, kwargs, false);
	}
	
	protected DLModelPytorchProtected(String modelFile, String callable, String importModule, String weightsPath, 
			Map<String, Object> kwargs, boolean customJDLL) throws IOException {
		if (!customJDLL && (new File(modelFile).isFile() == false || !modelFile.endsWith(".py")) && importModule == null)
			throw new IllegalArgumentException("The model file does not correspond to an existing .py file.");
		if (new File(weightsPath).isFile() == false 
				|| (!customJDLL 
						&& !(weightsPath.endsWith(".pt") || weightsPath.endsWith(".pth"))
						)
				)
			throw new IllegalArgumentException("The weights file does not correspond to an existing .pt/.pth file.");
		this.callable = callable;
		if (!customJDLL && (modelFile != null && new File(modelFile).isFile()))
			this.modelFile = new File(modelFile).getAbsolutePath();
		else 
			this.modelFile = null;
		if (!customJDLL && importModule != null)
			this.importModule = importModule;
		else 
			this.importModule = null;
		if (new File(weightsPath).isFile())
			this.modelFolder = new File(weightsPath).getParentFile().getAbsolutePath();
		else if (new File(modelFile).isFile())
			this.modelFolder = new File(modelFile).getParentFile().getAbsolutePath();
		this.weightsPath = new File(weightsPath).getAbsolutePath();
		this.kwargs = kwargs;
		this.envPath = INSTALLATION_DIR + File.separator + "envs" + File.separator + COMMON_PYTORCH_ENV_NAME;
		createPythonService();
	}
	
	protected void createPythonService() throws IOException {
		Environment env = new Environment() {
			@Override public String base() { return envPath; }
			};
		python = env.python();
		python.debug(System.err::println);
	}
	
	public String getEnvPath() {
		return this.envPath;
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
		if (closed)
			throw new RuntimeException("Cannot load model after it has been closed");

		Task task;
		try {
			String code = buildModelCode();
			code += RECOVER_OUTPUTS_CODE;
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

    /**
     * Copies the file from `inputPath` to `outputPath`, replacing every '+' with 'JDLL'.
     */
    private static void copyAndReplace(String inputPath, String outputPath) throws IOException {
    	if (new File(outputPath).isFile())
    		return;
        Files.write(Paths.get(outputPath), Files.readAllBytes(Paths.get(inputPath)));
    }
	
	protected String buildModelCode() throws IOException {
		String addPath = "";
		String importStr = "";
		String code = ""
				+ "device = 'cpu'" + System.lineSeparator()
				+ "if 'torch' not in globals().keys():" + System.lineSeparator()
				+ "  import torch" + System.lineSeparator()
				+ "  globals()['torch'] = torch" + System.lineSeparator()
				+ (!IS_ARM ? "" 
						: "  if torch.backends.mps.is_built() and torch.backends.mps.is_available():" + System.lineSeparator()
						+ "    device = 'mps'" + System.lineSeparator())
				+ "globals()['device'] = device" + System.lineSeparator();
		if (modelFile != null) {
			String moduleName = new File(modelFile).getName();
			moduleName = moduleName.substring(0, moduleName.length() - 3);
			if (moduleName.contains("+")) {
				String newModelFile = modelFile.replaceAll("\\+", JDLL_UUID);
				copyAndReplace(modelFile, newModelFile);
				moduleName = new File(newModelFile).getName();
				moduleName = moduleName.substring(0, moduleName.length() - 3);
				addPath = String.format("sys.path.append(os.path.abspath(r'%s'))", new File(newModelFile).getParentFile().getAbsolutePath());
				importStr = String.format("from %s import %s", moduleName, callable);
			} else {
				addPath = String.format("sys.path.append(os.path.abspath(r'%s'))", new File(modelFile).getParentFile().getAbsolutePath());
				importStr = String.format("from %s import %s", moduleName, callable);
			}
		} else {
			importStr = String.format("from %s import %s", this.importModule, callable);
		}
		code += String.format(LOAD_MODEL_CODE_ABSTRACT, addPath, importStr, callable, callable, callable);
		
		code += MODEL_VAR_NAME + "=" + callable + "(" + codeForKwargs()  + ")" + System.lineSeparator();
		code += "if any(isinstance(m, torch.nn.ConvTranspose3d) for m in " + MODEL_VAR_NAME + ".modules()):" + System.lineSeparator();
		code += "  device = 'cpu'" + System.lineSeparator();
		code += MODEL_VAR_NAME + ".to(device)" + System.lineSeparator();
		code += "try:" + System.lineSeparator()
				+ "  " + MODEL_VAR_NAME + ".load_state_dict("
				+ "torch.load(r'" + this.weightsPath + "', map_location=" + MODEL_VAR_NAME  + ".device))" + System.lineSeparator()
				+ "except:" + System.lineSeparator()
				+ "  " + MODEL_VAR_NAME + ".load_state_dict("
				+ "torch.load(r'" + this.weightsPath + "', map_location=torch.device(device)))" + System.lineSeparator();
		code += "globals()['" + MODEL_VAR_NAME + "'] = " + MODEL_VAR_NAME + System.lineSeparator();
		return code;
	}
	
	private String codeForKwargsList(List<Object> list) {
		String code = "[";
		for (Object codeVal : list) {
			if (codeVal == null)
				code += "None";
			else if ((codeVal instanceof Boolean && (Boolean) codeVal) || codeVal.equals("true"))
				code += "True";
			else if ((codeVal instanceof Boolean && !((Boolean) codeVal)) || codeVal.equals("false"))
				code += "False";
			else if (codeVal instanceof String)
				code += "\"" + codeVal + "\"";
			else if (codeVal instanceof List)
				code += codeForKwargsList((List<Object>) codeVal);
			else if (codeVal instanceof Map)
				code += codeForKwargsMap((Map<String, Object>) codeVal);
			else
				code += codeVal;
			code += ",";
		}
		code += "]";
		return code;
	}
	
	private String codeForKwargsMap(Map<String, Object> map) {
		String code = "{";
		for (Entry<String, Object> entry : map.entrySet()) {
			Object codeVal = entry.getValue();
			code += "'" + entry.getKey() + "':";
			if (codeVal == null)
				code += "None";
			else if ((codeVal instanceof Boolean && (Boolean) codeVal) || codeVal.equals("true"))
				code += "True";
			else if ((codeVal instanceof Boolean && !((Boolean) codeVal)) || codeVal.equals("false"))
				code += "False";
			else if (codeVal instanceof String)
				code += "\"" + codeVal + "\"";
			else if (codeVal instanceof List)
				code += codeForKwargsList((List<Object>) codeVal);
			else if (codeVal instanceof Map)
				code += codeForKwargsMap((Map<String, Object>) codeVal);
			else
				code += codeVal;
			code += ",";
		}
		code += "}";
		return code;
	}
	
	private String codeForKwargs() {
		String code = "";
		for (Entry<String, Object> ee : kwargs.entrySet()) {
			Object codeVal = ee.getValue();
			if (codeVal == null)
				codeVal = "None";
			else if ((codeVal instanceof Boolean && (Boolean) codeVal) || codeVal.equals("true"))
				codeVal = "True";
			else if ((codeVal instanceof Boolean && !((Boolean) codeVal)) || codeVal.equals("false"))
				codeVal = "False";
			else if (codeVal instanceof String)
				codeVal = "\"" + codeVal + "\"";
			else if (codeVal instanceof List)
				codeVal = codeForKwargsList((List<Object>) codeVal);
			else if (codeVal instanceof Map)
				codeVal = codeForKwargsMap((Map<String, Object>) codeVal);
			code += ee.getKey() + "=" + codeVal + ",";
		}
		return code;
	}

	@Override
	public void close() {
		if (!loaded)
			return;
		python.close();
		loaded = false;
		closed = true;
		
	}
	
	private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	Map<String, RandomAccessibleInterval<R>> predictForInputTensors(List<Tensor<T>> inTensors) 
	throws RunModelException {
		if (!loaded)
			throw new RuntimeException("Please load the model first.");
		List<String> names = inTensors.stream()
				.map(tt -> tt.getName() + "_np").collect(Collectors.toList());
		List<RandomAccessibleInterval<T>> rais = inTensors.stream().map(tt -> tt.getData()).collect(Collectors.toList());
		return executeCode(createInputsCode(rais, names));		
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
		List<String> names = IntStream.range(0, inputs.size())
				.mapToObj(i -> "var_" + UUID.randomUUID().toString().replace("-", "_")).collect(Collectors.toList());
		String code = createInputsCode(inputs, names);
		Map<String, RandomAccessibleInterval<R>> map = executeCode(code);
		List<RandomAccessibleInterval<R>> outRais = new ArrayList<RandomAccessibleInterval<R>>();
		for (Entry<String, RandomAccessibleInterval<R>> ee : map.entrySet()) {
			outRais.add(ee.getValue());
		}
		return outRais;
	}
	
	protected <T extends RealType<T> & NativeType<T>> String createInputsCode(List<RandomAccessibleInterval<T>> rais, List<String> names) {
		String code = "created_shms = []" + System.lineSeparator();
		code += "try:" + System.lineSeparator();
		for (int i = 0; i < rais.size(); i ++) {
			SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(rais.get(i), false, false);
			code += codeToConvertShmaToPython(shma, names.get(i));
			inShmaList.add(shma);
		}
		code += "  " + MODEL_VAR_NAME + ".eval()" + System.lineSeparator();
		code += "  with torch.no_grad():" + System.lineSeparator();
		code += "    " + OUTPUT_LIST_KEY + " = " + MODEL_VAR_NAME + "(";
		for (int i = 0; i < rais.size(); i ++)
			code += "torch.from_numpy(" + names.get(i) + ").to(device), ";
		code = code.substring(0, code.length() - 2);
		code += ")" + System.lineSeparator();
		code += ""
				+ "  " + SHMS_KEY + " = []" + System.lineSeparator()
				+ "  " + SHM_NAMES_KEY + " = []" + System.lineSeparator()
				+ "  " + DTYPES_KEY + " = []" + System.lineSeparator()
				+ "  " + DIMS_KEY + " = []" + System.lineSeparator()
				+ "  " + "globals()['" + SHMS_KEY + "'] = " + SHMS_KEY + System.lineSeparator()
				+ "  " + "globals()['" + SHM_NAMES_KEY + "'] = " + SHM_NAMES_KEY + System.lineSeparator()
				+ "  " + "globals()['" + DTYPES_KEY + "'] = " + DTYPES_KEY + System.lineSeparator()
				+ "  " + "globals()['" + DIMS_KEY + "'] = " + DIMS_KEY + System.lineSeparator();
		code += "  " + "handle_output_list(" + OUTPUT_LIST_KEY + ")" + System.lineSeparator();
		String closeEverythingWin = closeSHMWin();
		code += "  " + closeEverythingWin + System.lineSeparator();
		code += "except Exception as e:" + System.lineSeparator();
		code += "  " + closeEverythingWin + System.lineSeparator();
		code += "  raise e" + System.lineSeparator();
		code += taskOutputsCode();
		return code;
	}
	
	private static String closeSHMWin() {
		if (!PlatformDetection.isWindows())
			return "";
		return "[(shm_i.close(), shm_i.unlink()) for shm_i in created_shms]";
	}
	
	protected String taskOutputsCode() {
		String code = ""
				+ "task.outputs['" + SHM_NAMES_KEY + "'] = " + SHM_NAMES_KEY + System.lineSeparator()
				+ "task.outputs['" + DTYPES_KEY + "'] = " + DTYPES_KEY + System.lineSeparator()
				+ "task.outputs['" + DIMS_KEY + "'] = " + DIMS_KEY + System.lineSeparator();
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

	protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
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
			if (elem instanceof Object[] == false && elem instanceof List == false)
				throw new RuntimeException("Unexpected type for element of  '" + DIMS_KEY + "' list.");
			if (elem instanceof Object[]) {
				Object[] arr = (Object[]) elem;
				long[] longArr = new long[arr.length];
				for (int i = 0; i < arr.length; i ++) {
					if (arr[i] instanceof Number == false)
						throw new RuntimeException("Unexpected type for array of element of  '" + DIMS_KEY + "' list.");
					longArr[i] = ((Number) arr[i]).longValue();
				}
				outShmDims.add(longArr);
			} else if (elem instanceof List) {
				@SuppressWarnings("unchecked")
				List<Object> arr = (List<Object>) elem;
				long[] longArr = new long[arr.size()];
				for (int i = 0; i < arr.size(); i ++) {
					if (arr.get(i) instanceof Number == false)
						throw new RuntimeException("Unexpected type for array of element of  '" + DIMS_KEY + "' list.");
					longArr[i] = ((Number) arr.get(i)).longValue();
				}
				outShmDims.add(longArr);
			} else {
				throw new RuntimeException("Unexpected type for element of  '" + DIMS_KEY + "' list.");
			}
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
	
	protected static String codeToConvertShmaToPython(SharedMemoryArray shma, String varName) {
		String code = "";
		// This line wants to recreate the original numpy array. Should look like:
		// input0_appose_shm = shared_memory.SharedMemory(name=input0)
		// input0 = np.ndarray(size, dtype="float64", buffer=input0_appose_shm.buf).reshape([64, 64])
		code += "  " + varName + "_shm = shared_memory.SharedMemory(name='"
							+ shma.getNameForPython() + "', size=" + shma.getSize() 
							+ ")" + System.lineSeparator();
		code += "  " + "created_shms.append(" + varName + "_shm)" + System.lineSeparator();
		long nElems = 1;
		for (long elem : shma.getOriginalShape()) nElems *= elem;
		code += "  " + varName + " = np.ndarray(" + nElems  + ", dtype='" + CommonUtils.getDataTypeFromRAI(Cast.unchecked(shma.getSharedRAI()))
			  + "', buffer=" + varName +"_shm.buf).reshape([";
		for (int i = 0; i < shma.getOriginalShape().length; i ++)
			code += shma.getOriginalShape()[i] + ", ";
		code += "])" + System.lineSeparator();
		return code;
	}
	
	/**
	 * Check whether everything that is needed for Stardist 2D is installed or not
	 * @return true if the full python environment is installed or not
	 */
	public static boolean isInstalled() {
		return isInstalled(null);
	}
	
	/**
	 * Check whether everything that is needed for Stardist 2D is installed or not
	 * @return true if the full python environment is installed or not
	 */
	public static boolean isInstalled(String envPath) {
		if (envPath == null)
			envPath = COMMON_PYTORCH_ENV_NAME;
		
		Mamba mamba = new Mamba(INSTALLATION_DIR);
		try {
			 boolean inst = mamba.checkAllDependenciesInEnv(envPath, BIAPY_CONDA_DEPS);
			 if (!inst) return inst;
			 inst = mamba.checkAllDependenciesInEnv(envPath, BIAPY_PIP_DEPS_TORCH);
			 if (!inst) return inst;
			 inst = mamba.checkAllDependenciesInEnv(envPath, BIAPY_PIP_DEPS);
			 if (!inst) return inst;
		} catch (MambaInstallException e) {
			return false;
		}
		return true;
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
			biapyPythonInstalled = mamba.checkAllDependenciesInEnv(COMMON_PYTORCH_ENV_NAME, BIAPY_PIP_DEPS_TORCH);
			biapyPythonInstalled = mamba.checkAllDependenciesInEnv(COMMON_PYTORCH_ENV_NAME, BIAPY_PIP_DEPS);
			if (PlatformDetection.isMacOS() && PlatformDetection.getOSVersion().getMajor() < 14)
				biapyPythonInstalled = mamba.checkDependencyInEnv(COMMON_PYTORCH_ENV_NAME, "biapy==3.5.10");
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
			if (PlatformDetection.isMacOS() && PlatformDetection.getOSVersion().getMajor() < 14)
				mamba.pipInstallIn(COMMON_PYTORCH_ENV_NAME, new String[] {"biapy==3.5.10", "--no-deps"});
		};
		
		if (!isInstalled(INSTALLATION_DIR))
			throw new RuntimeException("Not all the requried packages were installed correctly. Please try again."
					+ " If the error persists, please post an issue at: https://github.com/bioimage-io/JDLL/issues");
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
