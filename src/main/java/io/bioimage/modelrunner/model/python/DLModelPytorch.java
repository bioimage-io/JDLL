package io.bioimage.modelrunner.model.python;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.function.Consumer;

import org.apache.commons.compress.archivers.ArchiveException;

import io.bioimage.modelrunner.apposed.appose.Environment;
import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.apposed.appose.Service;
import io.bioimage.modelrunner.apposed.appose.Service.Task;
import io.bioimage.modelrunner.apposed.appose.Service.TaskStatus;
import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.BaseModel;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;

public class DLModelPytorch extends BaseModel {
	
	protected final String modelFile;
	
	protected final String callable;
	
	protected final String weightsPath;
	
	protected final Map<String, Object> kwargs;
	
	public String envPath;
	
	private Service python;
	
	private List<SharedMemoryArray> inShmaList = new ArrayList<SharedMemoryArray>();
	
	public static final String COMMON_PYTORCH_ENV_NAME = "biapy";
	
	private static final List<String> BIAPY_CONDA_DEPS = Arrays.asList(new String[] {"python=3.10"});
	
	private static final List<String> BIAPY_PIP_DEPS = Arrays.asList(new String[] {"python=3.10", 
			"torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0",
			"timm==1.0.14", "pytorch-msssim==1.0.0", "torchmetrics[image]==1.4.*",
			"biapy==3.5.10", "appose",
			"--index-url https://download.pytorch.org/whl/cpu"});
		
	private static String INSTALLATION_DIR = Mamba.BASE_PATH;
	
	private static final String MODEL_VAR_NAME = "model_" + UUID.randomUUID().toString();

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
	
	private DLModelPytorch(String modelFile, String callable, String weightsPath, 
			Map<String, Object> kwargs) throws IOException {
		if (new File(modelFile).isFile() == false || !modelFile.endsWith(".py"))
			throw new IllegalArgumentException("The model file does not correspond to an existing .py file.");
		if (new File(weightsPath).isFile() == false || (!modelFile.endsWith(".pt") && !modelFile.endsWith(".pth")))
			throw new IllegalArgumentException("The weights file does not correspond to an existing .pt/.pth file.");
		this.callable = callable;
		this.modelFile = new File(modelFile).getAbsolutePath();
		this.weightsPath = new File(weightsPath).getAbsolutePath();
		this.kwargs = kwargs;
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

	@Override
	public void loadModel() throws LoadModelException {
		if (loaded)
			return;
		String moduleName = new File(modelFile).getName();
		moduleName = moduleName.substring(0, moduleName.length() - 3);
		String code = String.format(LOAD_MODEL_CODE_ABSTRACT, this.modelFile, moduleName, callable);
		code += buildModelCode();

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

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> void run(
			List<Tensor<T>> inTensors, List<Tensor<R>> outTensors) throws RunModelException {
		String code = "";
		for (Tensor<T> in : inTensors) {
			SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(in.getData());
			code += codeToConvertShmaToPython(shma, in.getName() + "_torch");
			inShmaList.add(shma);
		}
		for (Tensor<R> out : outTensors)
			code += out.getName() + ", ";
		code = code.substring(0, code.length() - 2);
		code += " = " + MODEL_VAR_NAME + "(";
		for (Tensor<T> in : inTensors)
			code += in.getName() + ", ";
		code = code.substring(0, code.length() - 2);
		code += ")" + System.lineSeparator();
		
	}

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> List<Tensor<T>> run(
			List<Tensor<R>> inputTensors) throws RunModelException {
		// TODO Auto-generated method stub
		return null;
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
