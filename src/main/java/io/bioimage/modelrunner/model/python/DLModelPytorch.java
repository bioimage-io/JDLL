package io.bioimage.modelrunner.model.python;

import java.io.File;
import java.util.List;
import java.util.Map;

import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.BaseModel;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class DLModelPytorch extends BaseModel {
	
	private static String INSTALLATION_DIR = 
			(System.getProperty("user.home") == null || System.getProperty("user.home").equals("")) 
			? Mamba.BASE_PATH : System.getProperty("user.home") + File.separator + ".jdll";
	
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
	
	public static boolean isInstalled() {
		return true;
	}

}
