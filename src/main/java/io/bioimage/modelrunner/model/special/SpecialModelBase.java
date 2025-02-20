package io.bioimage.modelrunner.model.special;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import io.bioimage.modelrunner.apposed.appose.Environment;
import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.apposed.appose.Service;
import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.apposed.appose.Service.Task;
import io.bioimage.modelrunner.apposed.appose.Service.TaskStatus;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.model.BaseModel;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;

public abstract class SpecialModelBase extends BaseModel {
	
	protected String envPath;
	
	protected ModelDescriptor descriptor;
		
	protected Service python;
	
	protected static String INSTALLATION_DIR = Mamba.BASE_PATH;
	
	protected abstract String createImportsCode();
	
	protected abstract <T extends RealType<T> & NativeType<T>>  void checkInput(RandomAccessibleInterval<T> image);
	
	protected abstract <T extends RealType<T> & NativeType<T>> 
	Map<String, RandomAccessibleInterval<T>> reconstructOutputs(Task task) throws IOException;

	public abstract <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	Map<String, RandomAccessibleInterval<R>> run(RandomAccessibleInterval<T> img) 
			throws IOException, InterruptedException;
	
	
	protected void createPythonService() throws IOException {
		Environment env = new Environment() {
			@Override public String base() { return envPath; }
			};
		python = env.python();
		python.debug(System.err::println);
	}
	
	@Override
	public void close() {
		if (!loaded)
			return;
		python.close();
	}
	
	@Override
	public void loadModel() throws LoadModelException {
		String code = "";
		if (!loaded) {
			code += createImportsCode() + System.lineSeparator();
		}
		
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
			loaded = true;
		} catch (IOException | InterruptedException e) {
			throw new LoadModelException(Types.stackTrace(e));
		}
	}
	
	protected static <T extends RealType<T> & NativeType<T>> 
	RandomAccessibleInterval<T> copy(RandomAccessibleInterval<T> im) {
		return Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(im), 
				Util.getTypeFromInterval(Cast.unchecked(im)));
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
		code += varName + " = np.ndarray(" + nElems  + ", dtype='" + CommonUtils.getDataTypeFromRAI(Cast.unchecked(shma.getSharedRAI()))
			  + "', buffer=" + varName +"_shm.buf).reshape([";
		for (int i = 0; i < shma.getOriginalShape().length; i ++)
			code += shma.getOriginalShape()[i] + ", ";
		code += "])" + System.lineSeparator();
		return code;
	}
	
	/**
	 * Set the directory where the StarDist Python environment will be installed
	 * @param installationDir
	 * 	directory where the StarDist Python environment will be created
	 */
	public static void setInstallationDir(String installationDir) {
		INSTALLATION_DIR = installationDir;
	}
	
	/**
	 * 
	 * @return the directory where the StarDist Python environment will be created
	 */
	public static String getInstallationDir() {
		return INSTALLATION_DIR;
	}
}
