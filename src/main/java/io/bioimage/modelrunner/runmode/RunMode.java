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
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.stream.IntStream;

import org.apposed.appose.Appose;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;

import io.bioimage.modelrunner.runmode.ops.OpInterface;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.SharedMemoryArray;
import io.bioimage.modelrunner.tensor.SharedMemoryFile;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

import java.util.UUID;

public class RunMode {
	
	private static final String IMPORT_XARRAY = "import xarray as xr" + System.lineSeparator(); 
	
	private static final String IMPORT_NUMPY = "import numpy as np" + System.lineSeparator();
	
	private static final String IMPORT_SHM = "from multiprocessing import shared_memory" + System.lineSeparator();
	
	protected static final String APPOSE_SHM_KEY = ("_shm_" + UUID.randomUUID().toString()).replace("-", "_");
	
	// TODO add support for list of objects
	private static final String OUTPUT_REFORMATING = ""
			+ "if isinstance(%s, xr.DataArray) and True:" + System.lineSeparator()
			+ "  %s = " + RunModeScripts.XR_METHOD + "_file(%s)" + System.lineSeparator()
			+ "elif isinstance(%s, xr.DataArray):" + System.lineSeparator()
			+ "  %s = " + RunModeScripts.XR_METHOD + "(%s)" + System.lineSeparator()
			+ "elif isinstance(%s, np.ndarray):" + System.lineSeparator()
			+ "  %s = " + RunModeScripts.NP_METHOD + "(%s)" + System.lineSeparator()
			+ "elif isinstance(%s, list):" + System.lineSeparator()
			+ "  %s = " + RunModeScripts.LIST_METHOD + "(%s)" + System.lineSeparator()
			+ "elif isinstance(%s, dict):" + System.lineSeparator()
			+ "  %s = " + RunModeScripts.DICT_METHOD + "(%s)" + System.lineSeparator();
	
	private static final String DEFAULT_IMPORT = 
			"import sys" + System.lineSeparator()
			+ IMPORT_NUMPY;
	
	
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
	private String shmInstancesCode = "shm_in_list = []" + System.lineSeparator();
	private String closeShmCode = "";
	private String moduleName;
	private List<SharedMemoryArray> shmaList = new ArrayList<SharedMemoryArray>();
	private List<String> outputNames = new ArrayList<String>();
	private List<String> filesToDestroy = new ArrayList<String>();
	
	private static String CLOSE_IN_SHMA = "for shm in shm_in_list:" + System.lineSeparator()
										+ "  shm.close()" + System.lineSeparator()
										+ "  shm.unlink()" + System.lineSeparator();
	
	private RunMode(OpInterface op) throws Exception {
		this.op = op;
		this.moduleName = op.getOpPythonFilename().substring(0, op.getOpPythonFilename().length() - 3);
		IntStream.range(0, op.getNumberOfOutputs()).forEach(i -> outputNames.add("output" + i));
		addImports();
		try {
			convertInputMap();
		} catch (Exception e) {
			throw new Exception("Error unpacking the Java inputs into the Python Appose process.", e);
		}
		opExecutionCode();
		retrieveResultsCode();
		
		opCode = shmInstancesCode + System.lineSeparator()
				+ importsCode + System.lineSeparator()
				+ RunModeScripts.TYPE_CONVERSION_METHODS_SCRIPT + System.lineSeparator()
				+ tensorRecreationCode + System.lineSeparator()
				+ opMethodCode + System.lineSeparator()
				+ retrieveResultsCode + System.lineSeparator()
				+ taskOutputCode;
		System.out.println(opCode);
	}
	
	public static RunMode createRunMode(OpInterface op) throws Exception {
		return new RunMode(op);
	}
	
	public Map<String, Object> runOP() {
		env = new Environment() {
			@Override public String base() { return op.getCondaEnv(); }
			@Override public boolean useSystemPath() { return false; }
			};
		Map<String, Object> outputs = null;
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
            outputs = recreateOutputObjects(task.outputs);
            Task closeShmTask = python.task(closeShmCode, null);
            closeShmTask.waitFor();
        } catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		this.shmaList.stream().forEach(entry ->entry.close());
		return outputs;
	}
	
	@SuppressWarnings("unchecked")
	private static Map<String, Object> recreateOutputObjects(Map<String, Object> apposeOuts) throws FileNotFoundException, IOException {
		 LinkedHashMap<String, Object> jdllOuts = new LinkedHashMap<String, Object>();
		 for (Entry<String, Object> entry : apposeOuts.entrySet()) {
			 Object value = entry.getValue();
			 
			 if (value instanceof Map && ((Map) value).get(RunModeScripts.APPOSE_DT_KEY) != null
					 && ((Map<String, Object>) value).get(RunModeScripts.APPOSE_DT_KEY).equals(RunModeScripts.TENSOR_KEY) ) {
				 if (((Map<String, Object>) value).get(RunModeScripts.NAME_KEY) == null)
					 ((Map<String, Object>) value).put(RunModeScripts.NAME_KEY, entry.getKey());
				 jdllOuts.put(entry.getKey(), createTensorFromApposeOutput((Map<String, Object>) value));
			 } else if (value instanceof Map && ((Map) value).get(RunModeScripts.APPOSE_DT_KEY) != null
					 && ((Map<String, Object>) value).get(RunModeScripts.APPOSE_DT_KEY).equals("tensor_file") ) {
				 if (((Map<String, Object>) value).get(RunModeScripts.NAME_KEY) == null)
					 ((Map<String, Object>) value).put(RunModeScripts.NAME_KEY, entry.getKey());
				 jdllOuts.put(entry.getKey(), createTensorFromApposeOutputFile((Map<String, Object>) value));
			 } else if (value instanceof Map && ((Map) value).get(RunModeScripts.APPOSE_DT_KEY) != null
					 && ((Map<String, Object>) value).get(RunModeScripts.APPOSE_DT_KEY).equals(RunModeScripts.NP_ARR_KEY) ) {
				 jdllOuts.put(entry.getKey(), createImgLib2ArrFromApposeOutput((Map<String, Object>) value));
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
		 return jdllOuts;
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
	
	private < T extends RealType< T > & NativeType< T > > void convertInputMap() throws Exception {
		apposeInputMap = new LinkedHashMap<>();
		if (op.getOpInputs() == null)
			return;
		for (Entry<String, Object> entry : this.op.getOpInputs().entrySet()) {
			if (entry.getValue() instanceof String) {
				apposeInputMap.put(entry.getKey(), entry.getValue());
			} else if (entry.getValue() instanceof Tensor && true) {
				String fileName = new File(UUID.randomUUID().toString() + ".npy").getAbsolutePath();
				SharedMemoryFile.buildFileFromRai(fileName, ((Tensor<T>) entry.getValue()).getData());
				filesToDestroy.add(fileName);
				apposeInputMap.put(entry.getKey(), null);
				addCodeToRecreateTensorFile(entry.getKey(), (Tensor<T>) entry.getValue(), fileName);
			} else if (entry.getValue() instanceof Tensor) {
				SharedMemoryArray shma = SharedMemoryArray.build(((Tensor<T>) entry.getValue()).getData());
				shmaList.add(shma);
				apposeInputMap.put(entry.getKey(), null);
				addCodeToRecreateTensor(entry.getKey(), shma, (Tensor<T>) entry.getValue());
			} else if (entry.getValue() instanceof RandomAccessibleInterval) {
				SharedMemoryArray shma = SharedMemoryArray.build((RandomAccessibleInterval<T>) entry.getValue());
				shmaList.add(shma);
				apposeInputMap.put(entry.getKey(), null);
				addCodeToRecreateNumpyArray(entry.getKey(), shma, (RandomAccessibleInterval<T>) entry.getValue());
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
				void addCodeToRecreateTensorFile(String ogName, Tensor<T> tensor, String filename) {
		if (!importsCode.contains(IMPORT_XARRAY))
			importsCode += IMPORT_XARRAY;
		if (!importsCode.contains(IMPORT_NUMPY))
			importsCode += IMPORT_NUMPY;
		// This line wants to recreate the original numpy array. Should look like:
		// input0 = xr.DataArray(np.array(input0).reshape([1, 1, 512, 512]), dims=["b", "c", "y", "x"], name="input0")
		this.tensorRecreationCode += ogName + " = xr.DataArray(np.load(" + filename + "), dims=[";
		for (String ss : tensor.getAxesOrderString().split(""))
			tensorRecreationCode += "\"" + ss + "\", ";
		tensorRecreationCode = 
				tensorRecreationCode.substring(0, tensorRecreationCode.length() - 2);
		tensorRecreationCode += "], name=\"" + tensor.getName() + "\")";
		tensorRecreationCode += System.lineSeparator();
	}
	
	private <T extends RealType<T> & NativeType<T>>
				void addCodeToRecreateTensor(String ogName, Tensor<T> tensor) {
		if (!importsCode.contains(IMPORT_XARRAY))
			importsCode += IMPORT_XARRAY;
		if (!importsCode.contains(IMPORT_NUMPY))
			importsCode += IMPORT_NUMPY;
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
	
	private <T extends RealType<T> & NativeType<T>>
				void addCodeToRecreateTensor(String ogName, SharedMemoryArray shma, Tensor<T> tensor) {
		if (!importsCode.contains(IMPORT_XARRAY))
			importsCode += IMPORT_XARRAY;
		if (!shmInstancesCode.contains(IMPORT_SHM))
			shmInstancesCode += IMPORT_SHM;
		// This line wants to recreate the original tensor array. Should look like:
		// input0_appose_shm = shared_memory.SharedMemory(name=input0)
		// input0 = xr.DataArray(np.ndarray(size, dtype="float64", 
		// 									buffer=input0_appose_shm.buf).reshape([64, 64]), 
		// 									dims=["b", "c", "y", "x"], name="input0")
		shmInstancesCode += ogName + APPOSE_SHM_KEY + " = shared_memory.SharedMemory(name='" 
						+ shma.getMemoryLocationPythonName() + "', size=" + shma.getSize() + ")" + System.lineSeparator();
		shmInstancesCode += "shm_in_list.append(" + ogName + APPOSE_SHM_KEY + ")" + System.lineSeparator();
		int size = 1;
		long[] dims = tensor.getData().dimensionsAsLongArray();
		for (long l : dims) {size *= l;}
		tensorRecreationCode += ogName + " = xr.DataArray(np.ndarray(" + size + ", dtype='" 
				+ CommonUtils.getDataType(tensor.getData()) + "', buffer=" 
				+ ogName + APPOSE_SHM_KEY + ".buf).reshape([";
		for (long ll : dims)
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
				void addCodeToRecreateNumpyArray(String ogName, SharedMemoryArray shma, RandomAccessibleInterval<T> rai) {
		if (!importsCode.contains(IMPORT_NUMPY))
			importsCode += IMPORT_NUMPY;
		if (!shmInstancesCode.contains(IMPORT_SHM))
			shmInstancesCode += IMPORT_SHM;
		// This line wants to recreate the original numpy array. Should look like:
		// input0_appose_shm = shared_memory.SharedMemory(name=input0)
		// input0 = np.ndarray(size, dtype="float64", buffer=input0_appose_shm.buf).reshape([64, 64])
		shmInstancesCode += ogName + APPOSE_SHM_KEY + " = shared_memory.SharedMemory(name='" 
							+ shma.getMemoryLocationPythonName() + "', size=" + shma.getSize() + ")" + System.lineSeparator();
		shmInstancesCode += "shm_in_list.append(" + ogName + APPOSE_SHM_KEY + ")" + System.lineSeparator();
		int size = 1;
		long[] dims = rai.dimensionsAsLongArray();
		for (long l : dims) {size *= l;}
		tensorRecreationCode += ogName + " = np.ndarray(" + size + ", dtype='" 
				+ CommonUtils.getDataType(rai) + "', buffer=" 
				+ ogName + APPOSE_SHM_KEY + ".buf).reshape([";
		for (long ll : dims)
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
			String code = String.format(OUTPUT_REFORMATING, outN, outN, outN, outN, outN, outN, outN, outN, outN,
					outN, outN, outN, outN, outN, outN);
			retrieveResultsCode += code;
			taskOutputCode += String.format("task.outputs['%s'] = %s", outN, outN)
					+ System.lineSeparator();
		}
	}
	
	private static < T extends RealType< T > & NativeType< T > > 
		Tensor<T> createTensorFromApposeOutput(Map<String, Object> apposeTensor) {
		String shmName = (String) apposeTensor.get(RunModeScripts.DATA_KEY);
		List<Integer> shape = (List<Integer>) apposeTensor.get(RunModeScripts.SHAPE_KEY);
		long[] longShape = new long[shape.size()];
		for (int i = 0; i < shape.size(); i ++) {longShape[i] = shape.get(i);}
		String dtype = (String) apposeTensor.get(RunModeScripts.DTYPE_KEY);
		String tensorname = (String) apposeTensor.get(RunModeScripts.NAME_KEY);
		String axes = (String) apposeTensor.get(RunModeScripts.AXES_KEY);
		boolean isFortran = (boolean) apposeTensor.get(RunModeScripts.IS_FORTRAN_KEY);
		RandomAccessibleInterval<T> rai = SharedMemoryArray.createImgLib2RaiFromSharedMemoryBlock(shmName, longShape, isFortran, dtype);
		return Tensor.build(tensorname, axes, rai);
	}
	
	private static < T extends RealType< T > & NativeType< T > > 
		Tensor<T> createTensorFromApposeOutputFile(Map<String, Object> apposeTensor) throws FileNotFoundException, IOException {
		String tensorname = (String) apposeTensor.get(RunModeScripts.NAME_KEY);
		String axes = (String) apposeTensor.get(RunModeScripts.AXES_KEY);
		String fileName = (String) apposeTensor.get("file_path");
		RandomAccessibleInterval<T> rai = SharedMemoryFile.buildRaiFromFile(fileName);
		return Tensor.build(tensorname, axes, rai);
	}
	
	private static < T extends RealType< T > & NativeType< T > > 
		RandomAccessibleInterval<T> createImgLib2ArrFromApposeOutput(Map<String, Object> apposeTensor) {
		String shmName = (String) apposeTensor.get(RunModeScripts.DATA_KEY);
		List<Integer> shape = (List<Integer>) apposeTensor.get(RunModeScripts.SHAPE_KEY);
		long[] longShape = new long[shape.size()];
		for (int i = 0; i < shape.size(); i ++) {longShape[i] = shape.get(i);}
		String dtype = (String) apposeTensor.get(RunModeScripts.DTYPE_KEY);
		boolean isFortran = (boolean) apposeTensor.get(RunModeScripts.IS_FORTRAN_KEY);
		RandomAccessibleInterval<T> rai = SharedMemoryArray.createImgLib2RaiFromSharedMemoryBlock(shmName, longShape, isFortran, dtype);
		return rai;
	}
	
	private static List<Object> createListFromApposeOutput(List<Object> list) throws FileNotFoundException, IOException {
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
