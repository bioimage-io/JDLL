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
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.stream.IntStream;

import org.apposed.appose.Appose;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;

import io.bioimage.modelrunner.runmode.ops.OpDescription;
import io.bioimage.modelrunner.tensor.ImgLib2ToArray;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class RunMode {
	
	private static final String IMPORT_XARRAY = "import xarray as xr" + System.lineSeparator(); 
	
	private static final String IMPORT_NUMPY = "import numpy as np" + System.lineSeparator(); 
	
	private static final String AXES_KEY = "axes";
	
	private static final String SHAPE_KEY = "shape";
	
	private static final String DATA_KEY = "data";
	
	private static final String NAME_KEY = "data";
	
	private static final String TENSOR_KEY = "tensor";
	
	private static final String NP_ARR_KEY = "np_arr";
	
	private static final String STANDARD_KEY = "standard";
	
	// TODO add support for list of objects
	private static final String OUTPUT_REFORMATING = 
			"if isinstance(%s, xr.DataArray):" + System.lineSeparator()
			+ "\ttypes_list.append(" + TENSOR_KEY + ")" + System.lineSeparator()
			+ "\t%s = {\"" + DATA_KEY + "\": %s.values.flatten().tolist(), \""
				+ SHAPE_KEY + "\": %s.shape, \"" + AXES_KEY + "\": %s.dims,"
				+ "\"" + NAME_KEY + "\": %s.name}" + System.lineSeparator()
			+ "elif isinstance(%s, np.ndarray):" + System.lineSeparator()
			+ "\ttypes_list.append(" + NP_ARR_KEY + ")" + System.lineSeparator()
			+ "\t%s = {\"" + DATA_KEY + "\": %s.flatten().tolist(), \""
			+ SHAPE_KEY + "\": %s.shape}" + System.lineSeparator()
			+ "elif isinstance(%s, list) and len(%s) == 0:" + System.lineSeparator()
			+ "\ttypes_list.append(" + STANDARD_KEY + ")" + System.lineSeparator()
			+ "elif isinstance(%s, list) and isinstance(%s[0], list):" + System.lineSeparator()
			+ "\ttypes_list.append(" + TENSOR_KEY + ")" + System.lineSeparator()
			+ "\t" + System.lineSeparator()
			+ "else:" + System.lineSeparator()
			+ "\ttypes_list.append(" + STANDARD_KEY + ")" + System.lineSeparator()
			+ "\t" + System.lineSeparator();
	
	private static final String BMZ_CORE_IMPORTS = 
			"from bioimageio.core import load_resource_description" + System.lineSeparator()
			+ "from bioimageio.core.resource_io.nodes import Model" + System.lineSeparator();
	private static final String OP_PACKAGE = "bioimageio.workflows";
	
	
	private Environment env;
	private String envFileName;
	private String opCode;
	private String opName;
	private String referencedModel;
	private OpDescription op;
	private LinkedHashMap<String, Object> kwargs;
	private LinkedHashMap<String, Object> apposeInputMap;
	private String tensorRecreationCode = "";
	private String importsCode = "";
	private String opMethodCode = "";
	private String retrieveResultsCode = "";
	List<String> outputNames = new ArrayList<String>();
	
	private RunMode(OpDescription op) {
		this.op = op;
		IntStream.range(0, op.getNumberOfOutputs()).forEach(i -> outputNames.add("output" + i));
		/*
		envFileName = op.getCondaEnv();
		referencedModel = op.appliedOnWhichModel();
		opName = op.getMethodName();
		kwargs = op.getMethodExtraArgs();
		*/
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
	
	private void addImports() {
		opCode += BMZ_CORE_IMPORTS + op.getOpImport() + System.lineSeparator();
	}
	
	private void opExecutionCode() {
		opMethodCode += op.getMethodName() + "(";
		for (String key : this.apposeInputMap.keySet())
			opMethodCode += key + ",";
		opMethodCode += ")" + System.lineSeparator();
	}
	
	private void retrieveResultsCode() {
		retrieveResultsCode = "task.update('Preparing outputs')" + System.lineSeparator();
		
		for (String outN : this.outputNames) {
			retrieveResultsCode += "if isinstance(" + outN + ", np.ndarray):" + System.lineSeparator()
								+ "\t"
								+ "elif isinstance(" + outN + ", np.ndarray):" + System.lineSeparator()
								+ "\t"
								+ ;
		}

		+ "\r\n"
		+ "if len(labels.shape) == 2:  # batch dim got squeezed\r\n"
		+ "    labels = labels[None]\r\n"
		+ "\r\n"
		+ "output_axes_wo_channels = tuple(a for a in model.outputs[0].axes if a != \"c\")\r\n"
		//+ "labels = labels.flatten().tolist()\r\n"
		+ "task.update('AAAAAAAAAAA')\r\n"
		+ "task.update(str(type(labels)))\r\n"
		+ "task.outputs['output0'] = labels\r\n";

	}
	
	private < T extends RealType< T > & NativeType< T > >
		void addCodeBody(List<Tensor<T>> inputTensors, List<Tensor<T>> outputTensors) {
		opCode += "\t" + "task.update('Start running workflow')" + System.lineSeparator();
		for (Tensor<T> output : outputTensors) 
			opCode += "\t" + output.getName() + ",";
		opCode = opCode.substring(0, opCode.length() - 1);
		opCode += " = await " + this.opName + "(" + referencedModel
			   + ",";
		for (Tensor<T> input : inputTensors) 
			opCode += input.getName() + ",";
		for (String key : kwargs.keySet())
			opCode += key + "=" + key + ",";
		opCode = opCode.substring(0, opCode.length() - 1);
		opCode += ")" + System.lineSeparator(); 
		opCode += "\t" + "task.update('Finished running workflow')" + System.lineSeparator();
		for (Tensor<T> output : outputTensors) 
			opCode += "\t" + "tasks['" + output.getName() + "'] = " + output.getName() + System.lineSeparator();
		opCode += "asyncio.run(run_workflow())" + System.lineSeparator();
	}
	
	public static void main(String[] args) {
		Integer[] arr = new Integer[3];
		arr[0] = 0;
		Object obj = (int) 2;
		boolean aa = isTypeDirectlySupported(obj.getClass());
		RunMode rm = new RunMode(null);

		rm.envFileName = "C:\\Users\\angel\\git\\jep\\miniconda\\envs\\stardist";
		rm.env = Appose.base(new File(rm.envFileName)).build();
		rm.referencedModel = "chatty-frog";
		rm.opName = "stardist_prediction_2d";
		rm.kwargs = new LinkedHashMap<String, Object>();
		
		final ImgFactory< FloatType > imgFactory = new CellImgFactory<>( new FloatType(), 5 );
		final Img< FloatType > img1 = imgFactory.create( 1, 512, 512, 3 );
		// Create the input tensor with the nameand axes given by the rdf.yaml file
		// and add it to the list of input tensors
		Tensor<FloatType> inpTensor = Tensor.build("input0", "byxc", img1);
		List<Tensor<FloatType>> inputs = new ArrayList<Tensor<FloatType>>();
		inputs.add(inpTensor);
		
		final Img< FloatType > img2 = imgFactory.create( 1, 512, 512, 1 );
		// Create the input tensor with the nameand axes given by the rdf.yaml file
		// and add it to the list of input tensors
		Tensor<FloatType> outTensor = Tensor.build("ouput0", "byxc", img2);
		List<Tensor<FloatType>> outputs = new ArrayList<Tensor<FloatType>>();
		outputs.add(outTensor);
		
		rm.run(inputs, outputs);
		
	}
	
	private < T extends RealType< T > & NativeType< T > > void convertInputMap() {
		apposeInputMap = new LinkedHashMap<>();
		for (Entry<String, Object> entry : this.kwargs.entrySet()) {
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
				HashMap<String, Object> tensorToMap(Tensor<T> tt) {
		HashMap<String, Object> tensorMap = new HashMap<String, Object>();
		tensorMap.put(AXES_KEY, tt.getAxesOrderString());
		tensorMap.put(DATA_KEY, ImgLib2ToArray.build(tt.getData()));
		tensorMap.put(SHAPE_KEY, tt.getShape());
		return tensorMap;
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
		tensorRecreationCode += "], name=" + tensor.getName() + "])";
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
		tensorRecreationCode += System.lineSeparator();
	}
	
	public < T extends RealType< T > & NativeType< T > >
		void run(List<Tensor<T>> inputTensors, List<Tensor<T>> outputTensors) {
		
		opCode = "";
		addImports();
		
		LinkedHashMap<String, Object> inputMap = new LinkedHashMap<>();
		for (Tensor<T> input : inputTensors) {
			HashMap<String, Object> tensorMap = new HashMap<String, Object>();
			tensorMap.put(AXES_KEY, input.getAxesOrderString());
			tensorMap.put(DATA_KEY, ImgLib2ToArray.build(input.getData()));
			tensorMap.put(SHAPE_KEY, input.getShape());
			inputMap.put(input.getName(), tensorMap);
			
			addRecreationOfTensor(input);
		}
		
		inputMap.putAll(kwargs);
		
		addCodeBody(inputTensors, outputTensors);
		opCode = "import numpy as np\r\n"
				+ "import asyncio\r\n"
				+ "import numpy as np\r\n"
				+ "import tempfile\r\n"
				+ "import warnings\r\n"
				+ "from math import ceil\r\n"
				+ "from os import PathLike\r\n"
				+ "from pathlib import Path\r\n"
				+ "from typing import Dict, IO, List, Optional, Tuple, Union\r\n"
				+ "\r\n"
				+ "import xarray as xr\r\n"
				+ "from stardist import import_bioimageio as stardist_import_bioimageio\r\n"
				+ "\r\n"
				+ "from bioimageio.core import export_resource_package, load_resource_description\r\n"
				+ "from bioimageio.core.prediction_pipeline._combined_processing import CombinedProcessing\r\n"
				+ "from bioimageio.core.prediction_pipeline._measure_groups import compute_measures\r\n"
				+ "from bioimageio.core.resource_io.nodes import Model\r\n"
				+ "from bioimageio.spec.model import raw_nodes\r\n"
				+ "from bioimageio.spec.shared.raw_nodes import ResourceDescription as RawResourceDescription\r\n"
				+ "from bioimageio.workflows import stardist_prediction_2d\r\n"
				+ "input_tensor = input0['data']\r\n"
				+ "input_tensor = np.array(input_tensor).reshape(1,512,512,3)\r\n"
				+ "model_rdf = 'chatty-frog'\r\n"
				+ "package_path = export_resource_package(model_rdf)\r\n"
				+ "with tempfile.TemporaryDirectory() as tmp_dir:\r\n"
				+ "    import_dir = Path(tmp_dir) / \"import_dir\"\r\n"
				+ "    imported_stardist_model = stardist_import_bioimageio(package_path, import_dir)\r\n"
				+ "\r\n"
				+ "model = load_resource_description(package_path)\r\n"
				+ "task.update('BBBBBBBBBBBB')\r\n"
				+ "labels, polys = imported_stardist_model.predict_instances(\r\n"
				+ "    input_tensor,\r\n"
				+ "    axes=\"\".join([{\"b\": \"S\"}.get(a[0], a[0].capitalize()) for a in model.inputs[0].axes]),\r\n"
				+ "    n_tiles=None,\r\n"
				+ ")\r\n"
				+ "\r\n"
				+ "if len(labels.shape) == 2:  # batch dim got squeezed\r\n"
				+ "    labels = labels[None]\r\n"
				+ "\r\n"
				+ "output_axes_wo_channels = tuple(a for a in model.outputs[0].axes if a != \"c\")\r\n"
				//+ "labels = labels.flatten().tolist()\r\n"
				+ "task.update('AAAAAAAAAAA')\r\n"
				+ "task.update(str(type(labels)))\r\n"
				+ "task.outputs['output0'] = labels\r\n";
        
        try (Service python = env.python()) {
        	python.debug(line -> {
        		System.err.println(line);
        	});
            Task task = python.task(opCode, inputMap);
            System.out.println("here");
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
                }
            });
            task.waitFor();
            System.out.println("here2");
            Object result = task.outputs.get("output0");
            Object polys = task.outputs.get("polys");
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
