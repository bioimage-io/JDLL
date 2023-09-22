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
package io.bioimage.modelrunner.runmode.ops;

import java.util.LinkedHashMap;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Code for the JDLL OP that allows running the whole stardist model (pre-processing + model executio
 * + post-processing and tiling) in Python using Appose
 * @author Carlos Javier Garcia Lopez de Haro
 *
 */
public class StardistInferJdllOp implements OpInterface {
	
	private String modelName;
	
	private Tensor inputTensor;
	
	private LinkedHashMap<String, Object> inputsMap;
	
	private final String CONDA_ENV_YAML_FILE = "";
	
	private final static String MODEL_KEY = "model";
	
	private final static String INPUT_TENSOR_KEY = "input_tensor";
	
	private static final String OP_METHOD_NAME = "stardist_prediction_2d_mine";
	
	private static final int N_STARDIST_OUTPUTS = 2;
	
	public void setModel(String modelName) {
		this.modelName = modelName;
	}
	
	public < T extends RealType< T > & NativeType< T > > void setInputTensor(Tensor<T> tensor) {
		inputTensor = tensor;
	}

	@Override
	public String getOpImport() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int getNumberOfOutputs() {
		return N_STARDIST_OUTPUTS;
	}

	@Override
	public void installOp() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LinkedHashMap<String, Object> getOpInputs() {
		inputsMap = new LinkedHashMap<String, Object>();
		inputsMap.put(MODEL_KEY, modelName);
		inputsMap.put(INPUT_TENSOR_KEY, inputTensor);
		return this.inputsMap;
	}

	@Override
	public String getCondaEnv() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getMethodName() {
		return OP_METHOD_NAME;
	}

	@Override
	public String getOpDir() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean isOpInstalled() {
		// TODO Auto-generated method stub
		return false;
	}
	
	public static boolean isModelCompatible() {
		
	}

}
