package io.bioimage.modelrunner.transformations;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;

import org.apposed.appose.Appose;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;

import io.bioimage.modelrunner.bioimageio.ops.OpDescription;
import io.bioimage.modelrunner.tensor.ImgLib2ToArray;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class RunMode {
	
	private static final String NAME_KEY = "name";
	
	private static final String AXES_KEY = "axes";
	
	private static final String DATA_KEY = "data";
	
	private static final String BMZ_CORE_IMPORTS = 
			"from bioimageio.core import load_resource_description" + System.lineSeparator()
			+ "from bioimageio.core.resource_io.nodes import Model" + System.lineSeparator();
	private static final String OP_PACKAGE = "bioimageio.workflows";
	
	private static final String[] RUN_MODES = 
			new String[] {"stardist", "cellpose", "mrcnn"};
	
	private Environment env;
	private String envFileName;
	private String opCode;
	
	private RunMode(OpDescription op) {
		envFileName = op.getCondaEnv();
	}
	
	public static RunMode createRunMode(OpDescription op) {
		return new RunMode(op);
	}
	
	public void envCreation() {
		if (checkRequiredEnvExists()) {
			env = Appose.base(new File(envFileName)).build();
			return;
		}
		env = Appose.conda(new File(envFileName)).build();
	}
	
	public boolean checkRequiredEnvExists() {
		return false;
	}
	
	private void createPythonScriptForOp() {
		opCode = "";
		addImports();
	}
	
	private void addImports() {
		opCode += "import " +  + System.lineSeparator();
	}
	
	public < T extends RealType< T > & NativeType< T > >
		void run(List<Tensor<T>> inputTensors, List<Tensor<T>> outputTensors) {
		LinkedHashMap<String, Object> inputMap = new LinkedHashMap<>();
		for (Tensor<T> input : inputTensors) {
			HashMap<String, Object> tensorMap = new HashMap<String, Object>();
			tensorMap.put(AXES_KEY, input.getAxesOrderString());
			tensorMap.put(DATA_KEY, ImgLib2ToArray.build(input.getData()));
			inputMap.put(NAME_KEY, tensorMap);
			inputMap.put("tensor", ImgLib2ToArray.buildFloat(input.getData()));
		}
        
        try (Service python = env.python()) {
        	python.debug(line -> {
        		System.err.println(line);
        	});
            Task task = python.task(CODE, inputMap);
            System.out.println("here");
            task.listen(event -> {
                switch (event.responseType) {
	                case UPDATE:
	                    System.out.println("Progress: " + task.message);
	                    break;
	                case COMPLETION:
	                    int numer = (Integer) task.outputs.get("value");
	                    System.out.println("Task complete. Result: " + numer);
	                    break;
	                case CANCELATION:
	                    System.out.println("Task canceled");
	                    break;
	                case FAILURE:
	                    System.out.println("Task failed: " + task.error);
	                    break;
                }
            });
            task.waitFor();
            System.out.println("here2");
            Object result = task.outputs.get("torch_sum_val");
            System.out.println("here3");
            if (result instanceof Integer)
            	System.out.print(result);
        } catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
