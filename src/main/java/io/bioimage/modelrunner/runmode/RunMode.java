/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.runmode;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.IntStream;

import org.apposed.appose.Appose;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;

import io.bioimage.modelrunner.runmode.ops.OpInterface;
import io.bioimage.modelrunner.tensor.ImgLib2ToArray;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class RunMode {
	
	private static final String IMPORT_XARRAY = "import xarray as xr" + System.lineSeparator(); 
	
	private static final String IMPORT_NUMPY = "import numpy as np" + System.lineSeparator();
	
	// TODO add support for list of objects
	private static final String OUTPUT_REFORMATING = 
			"if isinstance(%s, xr.DataArray):" + System.lineSeparator()
			+ "  task.update('is data array')" + System.lineSeparator()
			+ "  %s = " + RunModeScripts.XR_METHOD + "(%s)" + System.lineSeparator()
			+ "elif isinstance(%s, np.ndarray):" + System.lineSeparator()
			+ "  task.update('np array')" + System.lineSeparator()
			+ "  %s = " + RunModeScripts.NP_METHOD + "(%s)" + System.lineSeparator()
			+ "elif isinstance(%s, list):" + System.lineSeparator()
			+ "  task.update('is list')" + System.lineSeparator()
			+ "  %s = " + RunModeScripts.LIST_METHOD + "(%s)" + System.lineSeparator()
			+ "elif isinstance(%s, dict):" + System.lineSeparator()
			+ "  task.update('is dict')" + System.lineSeparator()
			+ "  task.update(str(output1))" + System.lineSeparator()
			+ "  %s = " + RunModeScripts.DICT_METHOD + "(%s)" + System.lineSeparator();
	
	private static final String DEFAULT_IMPORT = 
			"import sys" + System.lineSeparator();
	
	
	private Environment env;
	private String envFileName;
	private String opCode;
	private OpInterface op;
	private LinkedHashMap<String, Object> apposeInputMap;
	private String tensorRecreationCode = "";
	private String importsCode = "";
	private String opMethodCode = "";
	private String retrieveResultsCode = "";
	private String taskOutputCode = "";
	private String moduleName;
	List<String> outputNames = new ArrayList<String>();
	
	private RunMode(OpInterface op) {
		this.op = op;
		this.moduleName = op.getOpPythonFilename().substring(0, op.getOpPythonFilename().length() - 3);
		IntStream.range(0, op.getNumberOfOutputs()).forEach(i -> outputNames.add("output" + i));
		addImports();
		convertInputMap();
		opExecutionCode();
		retrieveResultsCode();
		
		opCode = importsCode + System.lineSeparator()
				+ RunModeScripts.TYPE_CONVERSION_METHODS_SCRIPT + System.lineSeparator()
				+ tensorRecreationCode + System.lineSeparator()
				+ opMethodCode + System.lineSeparator()
				+ retrieveResultsCode + System.lineSeparator()
				+ taskOutputCode;
		System.out.println(opCode);
	}
	
	public static RunMode createRunMode(OpInterface op) {
		return new RunMode(op);
	}
	
	public void testRunModel() {
		env = Appose.base(new File(this.op.getCondaEnv())).build();
		try (Service python = env.python()) {
        	python.debug(line -> {
        		System.err.println(line);
        	});
            Task task = python.task(opCode, apposeInputMap);
            task.listen(event -> {
                switch (event.responseType) {
	                case UPDATE:
	                    System.out.println("Progress: " + task.message);
	                    break;
	                case COMPLETION:
	                    Object numer =  task.outputs.get("output0");
	                    System.out.println("Task complete. Result: " + numer);
	                    break;
	                case CANCELATION:
	                    System.out.println("Task canceled");
	                    break;
	                case FAILURE:
	                    System.out.println("Task failed: " + task.error);
	                    break;
					case LAUNCH:
						System.out.println("LAunched code");
						break;
					default:
						break;
                }
            });
            task.waitFor();
            System.out.println("here2");
            Map<String, Object> aa = task.outputs;
            Object result = task.outputs.get("output0");
            Object polys = task.outputs.get("output1");
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
	
	private static Map<String, Object> recreateOutputObjects(Map<String, Object> apposeOuts) {
		 LinkedHashMap<String, Object> jdllOuts = new LinkedHashMap<String, Object>();
		 for (Entry<String, Object> entry : apposeOuts.entrySet()) {
			 Object value = entry.getValue();
			 
			 if (value instanceof Map && ((Map) value).get(RunModeScripts.APPOSE_DT_KEY) != null
					 && ((Map) value).get(RunModeScripts.APPOSE_DT_KEY).equals(RunModeScripts.TENSOR_KEY) ) {
				 
			 } else if (value instanceof Map && ((Map) value).get(RunModeScripts.APPOSE_DT_KEY) != null
					 && ((Map) value).get(RunModeScripts.APPOSE_DT_KEY).equals(RunModeScripts.NP_ARR_KEY) ) {
				 
			 } else if (value instanceof Map) {
				 jdllOuts.put(entry.getKey(), recreateOutputObjects((Map<String, Object>) value));
			 } else if (value instanceof List) {
				 jdllOuts.put(entry.getKey(), createListFromApposeOutput((List<Object>) value));
			 } else if (isTypeDirectlySupported(value.getClass())) {
				 jdllOuts.put(entry.getKey(), value);
			 } else {
				 throw new IllegalArgumentException("Type of output named: '" + entry.getKey()
				 							+ "' not supported (" + value.getClass() + ").");
			 }
		 }
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
	
	private void addImports() {
		importsCode = DEFAULT_IMPORT
				+ "sys.path.append(r'" + op.getOpDir() + "')" + System.lineSeparator()
				+ "import " + moduleName + System.lineSeparator()
				+ "task.update('Imports')" + System.lineSeparator();
	}
	
	private < T extends RealType< T > & NativeType< T > > void convertInputMap() {
		apposeInputMap = new LinkedHashMap<>();
		for (Entry<String, Object> entry : this.op.getOpInputs().entrySet()) {
			if (entry.getValue() instanceof String) {
				apposeInputMap.put(entry.getKey(), entry.getValue());
			} else if (entry.getValue() instanceof Tensor) {
				Object tensorArr = ImgLib2ToArray.build(((Tensor<T>) entry.getValue()).getData());
				apposeInputMap.put(entry.getKey(), tensorArr);
				addCodeToRecreateTensor(entry.getKey(), (Tensor<T>) entry.getValue());
			} else if (entry.getValue() instanceof RandomAccessibleInterval) {
				Object imgArr = ImgLib2ToArray.build(((RandomAccessibleInterval<T>) entry.getValue()));
				apposeInputMap.put(entry.getKey(), imgArr);
				addCodeToRecreateNumpyArray(entry.getKey(), (RandomAccessibleInterval<T>) entry.getValue());
			} else if (!entry.getValue().getClass().isArray() 
					&& isTypeDirectlySupported(entry.getValue().getClass())) {
				apposeInputMap.put(entry.getKey(), entry.getValue());
			} else if (entry.getValue().getClass().isArray() 
					&& isTypeDirectlySupported(entry.getValue().getClass().getComponentType())) {
				apposeInputMap.put(entry.getKey(), entry.getValue());
			} else if (entry.getValue() instanceof List 
					&& ((List) entry.getValue()).size() == 0) {
				apposeInputMap.put(entry.getKey(), new Object[0]);
			} else if (entry.getValue() instanceof List 
					&& isTypeDirectlySupported(((List) entry.getValue()).get(0).getClass())) {
				apposeInputMap.put(entry.getKey(), ((List) entry.getValue()).toArray());
			} else {
				throw new IllegalArgumentException("The type of the input argument: '"
						+ entry.getKey() + "' is not supported ("
						+ entry.getValue().getClass());
			}
		}
	}
	
	private static boolean isTypeDirectlySupported(Class<?> cl) {
		if (Number.class.isAssignableFrom(cl) 
				|| cl.isPrimitive() || String.class.isAssignableFrom(cl) ) {
			return true;
		}
		return false;
	}
	
	private <T extends RealType<T> & NativeType<T>>
				void addCodeToRecreateTensor(String ogName, Tensor<T> tensor) {
		if (!importsCode.contains(IMPORT_XARRAY))
			importsCode += IMPORT_XARRAY;
		if (!importsCode.contains(IMPORT_NUMPY))
			importsCode += IMPORT_NUMPY;
		tensorRecreationCode = "";
		tensorRecreationCode += "task.update('input conv')" + System.lineSeparator();
		// This line wants to recreate the original numpy array. Should look like:
		// input0 = xr.DataArray(np.array(input0).reshape([1, 1, 512, 512]), dims=["b", "c", "y", "x"], name="input0")
		this.tensorRecreationCode += ogName + " = xr.DataArray(np.array(" + ogName + ").reshape([";
		for (int ll : tensor.getShape())
			tensorRecreationCode += ll + ", ";
		tensorRecreationCode = 
				tensorRecreationCode.substring(0, tensorRecreationCode.length() - 2);
		tensorRecreationCode += "]), dims=[";
		for (String ss : tensor.getAxesOrderString().split(""))
			tensorRecreationCode += "\"" + ss + "\", ";
		tensorRecreationCode = 
				tensorRecreationCode.substring(0, tensorRecreationCode.length() - 2);
		tensorRecreationCode += "], name=\"" + tensor.getName() + "\")";
		tensorRecreationCode += System.lineSeparator();
	}
	
	private <T extends RealType<T> & NativeType<T>>
				void addCodeToRecreateNumpyArray(String ogName, RandomAccessibleInterval<T> rai) {
		if (!importsCode.contains(IMPORT_NUMPY))
			importsCode += IMPORT_NUMPY;
		// This line wants to recreate the original numpy array. Should look like:
		// np_arr = np.array(np_arr).reshape([1, 1, 512, 512])
		tensorRecreationCode += ogName + " = np.array(" + ogName + ").reshape([";
		for (long ll : rai.dimensionsAsLongArray())
			tensorRecreationCode += ll + ", ";
		tensorRecreationCode = 
				tensorRecreationCode.substring(0, tensorRecreationCode.length() - 2);
		tensorRecreationCode += "])" + System.lineSeparator();
	}
	
	private void opExecutionCode() {
		opMethodCode = "";
		opMethodCode += "task.update('method')" + System.lineSeparator();
		for (String outN : this.outputNames) {
			opMethodCode += outN + ", ";
		}
		opMethodCode = 
				opMethodCode.substring(0, opMethodCode.length() - 2);
		opMethodCode += " = ";
		opMethodCode += moduleName + "." + op.getMethodName() + "(";
		for (String key : this.apposeInputMap.keySet())
			opMethodCode += key + ",";
		opMethodCode += ")" + System.lineSeparator();
	}
	
	private void retrieveResultsCode() {
		retrieveResultsCode = "task.update('Preparing outputs')" + System.lineSeparator();
		
		for (String outN : this.outputNames) {
			String code = String.format(OUTPUT_REFORMATING, outN, outN, outN, outN, outN, outN,
					outN, outN, outN, outN, outN, outN, outN);
			retrieveResultsCode += code;
			taskOutputCode += String.format("task.outputs['%s'] = %s", outN, outN)
					+ System.lineSeparator();
		}
	}
	
	private static < T extends RealType< T > & NativeType< T > > 
		Tensor<T> createTensorFromApposeOutput(Map<String, Object> apposeTensor) {
		ArrayImgFactory<T> factory = new ArrayImgFactory<T>(new T());
		return null;
	}
	
	private static < T extends RealType< T > & NativeType< T > > 
		RandomAccessibleInterval<T> createImgLib2ArrFromApposeOutput(Map<String, Object> apposeTensor) {
		ArrayImgFactory<T> factory = new ArrayImgFactory<T>();
		return null;
	}
	
	private static List<Object> createListFromApposeOutput(List<Object> list) {
		List<Object> nList = new ArrayList<Object>();
		for (Object value : list) {
			 
			 if (value instanceof Map && ((Map) value).get(RunModeScripts.APPOSE_DT_KEY) != null
					 && ((Map) value).get(RunModeScripts.APPOSE_DT_KEY).equals(RunModeScripts.TENSOR_KEY) ) {
				 
			 } else if (value instanceof Map && ((Map) value).get(RunModeScripts.APPOSE_DT_KEY) != null
					 && ((Map) value).get(RunModeScripts.APPOSE_DT_KEY).equals(RunModeScripts.NP_ARR_KEY) ) {
				 
			 } else if (value instanceof Map) {
				 nList.add(recreateOutputObjects((Map<String, Object>) value));
			 } else if (value instanceof List) {
				 nList.add(createListFromApposeOutput((List<Object>) value));
			 } else if (isTypeDirectlySupported(value.getClass())) {
				 nList.add(value);
			 } else {
				 throw new IllegalArgumentException("Type of output"
				 							+ " not supported (" + value.getClass() + ").");
			 }
		 }
		return nList;
	}
}
