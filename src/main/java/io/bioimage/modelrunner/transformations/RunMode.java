package io.bioimage.modelrunner.transformations;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
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
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class RunMode {
	
	private static final String AXES_KEY = "axes";
	
	private static final String SHAPE_KEY = "shape";
	
	private static final String DATA_KEY = "data";
	
	private static final String BMZ_CORE_IMPORTS = 
			"import numpy as np" + System.lineSeparator()
			+ "import xarray as xr" + System.lineSeparator()
			+ "from bioimageio.core import load_resource_description" + System.lineSeparator()
			+ "from bioimageio.core.resource_io.nodes import Model" + System.lineSeparator();
	private static final String OP_PACKAGE = "bioimageio.workflows";
	
	
	private Environment env;
	private String envFileName;
	private String opCode;
	private String opName;
	private String referencedModel;
	private LinkedHashMap<String, Object> kwargs;
	
	private RunMode(OpDescription op) {
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
		opCode += BMZ_CORE_IMPORTS;
		opCode += "from " + OP_PACKAGE + " import " + opName + System.lineSeparator();
		opCode += "async def run_workflow():" + System.lineSeparator();
	}
	
	public < T extends RealType< T > & NativeType< T > >
		void addRecreationOfTensor(Tensor<T> tensor) {
		opCode += "\t" + tensor.getName() + "_data = " + tensor.getName() + "['" + DATA_KEY + "']" + System.lineSeparator()
				+ "\t" + tensor.getName() + " = np.array(" + tensor.getName() + "_data).reshape(";
		for (int i : tensor.getShape()) {
			opCode += i + ",";
		}
		opCode = opCode.substring(0, opCode.length() - 1);
		opCode += ")" + System.lineSeparator();
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