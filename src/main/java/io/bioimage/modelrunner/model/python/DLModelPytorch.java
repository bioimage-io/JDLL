package io.bioimage.modelrunner.model.python;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import org.apache.commons.compress.archivers.ArchiveException;

import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.BaseModel;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class DLModelPytorch extends BaseModel {
	
	public static final String COMMON_PYTORCH_ENV_NAME = "biapy";
	
	private static final List<String> BIAPY_CONDA_DEPS = Arrays.asList(new String[] {"python=3.10"});
	
	private static final List<String> BIAPY_PIP_DEPS = Arrays.asList(new String[] {"python=3.10", 
			"torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0",
			"timm==1.0.14", "pytorch-msssim==1.0.0", "torchmetrics[image]==1.4.*",
			"biapy==3.5.10", "appose",
			"--index-url https://download.pytorch.org/whl/cpu"});
		
	private static String INSTALLATION_DIR = Mamba.BASE_PATH;
	
	private DLModelPytorch(String modelClass, String callable, String modelPath, Map<String, Object> kwargs) {
		
	}
	
	public static DLModelPytorch create(String modelClass, String callable, String modelPath, Map<String, Object> kwargs) {
		return new DLModelPytorch(modelClass, callable, modelPath, kwargs);
	}

	@Override
	public void loadModel() throws LoadModelException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void close() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> void run(
			List<Tensor<T>> inTensors, List<Tensor<R>> outTensors) throws RunModelException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> List<Tensor<T>> run(
			List<Tensor<R>> inputTensors) throws RunModelException {
		// TODO Auto-generated method stub
		return null;
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
