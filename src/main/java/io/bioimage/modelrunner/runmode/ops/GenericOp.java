/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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

import java.io.File;
import java.util.LinkedHashMap;

import io.bioimage.modelrunner.runmode.RunMode;

/**
 * A generic OP that allows running Python code from JDLL if it is formatted correctly.
 * 
 * A good example of a GEneric OP can be seen with the stardist model.
 * TODO add meaningfill examples with details
 * @author Carlos Javier Garcia Lopez de Haro
 *
 */
public class GenericOp implements OpInterface {
	
	private String opFilePath;
	
	private String envPath;
	
	private String methodName;
	
	private int nOutputs;
	
	private LinkedHashMap<String, Object> inputsMap;
	
	/**
	 * Create the corresponding Python OP to be able to execute Python pre-designed scripts that 
	 * enable more complex tasks. One example is StarDist post-processing or cell-pose model 
	 * @param envYaml
	 * 	path to the Python Conda/Mamba environment yaml file that the
	 * 	OP needs to run properly
	 * @param script
	 * 	the path to the script that contains the wanted OP
	 * @param method
	 * 	the method of the script that needs to be called to run the OP
	 * @param nOutputs
	 * 	the final number of outputs that the method that runs the OP defined by the argument 'method' has
	 * @return the OP that can be used by the class {@link RunMode} to run
	 */
	public static GenericOp create(String envYaml, String script, String method, int nOutputs) {
		GenericOp op = new GenericOp();
		op.envPath = envYaml;
		op.opFilePath = script;
		op.methodName = method;
		op.nOutputs = nOutputs;
		return op;
	}
	
	/**
	 * Set the inputs to the OP. These are the inputs to the method that is used to run the OP in the OP script.
	 * The Map keys will be the names of the corresponding variables in Python and the values will be the variables values.
	 * @param kwargs
	 * 	Map containing the variables that will be the inputs to the OP main method in the Python script
	 */
	public void setInputs(LinkedHashMap<String, Object> kwargs) {
		this.inputsMap = kwargs;
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public String getOpPythonFilename() {
		return new File(this.opFilePath).getName();
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public int getNumberOfOutputs() {
		return nOutputs;
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public void installOp() {
		// TODO 
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public LinkedHashMap<String, Object> getOpInputs() {
		return this.inputsMap;
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public String getCondaEnv() {
		return envPath;
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public String getMethodName() {
		return this.methodName;
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public String getOpDir() {
		return new File(opFilePath).getParent();
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public boolean isOpInstalled() {
		// TODO maybe remove this method? Make the check at installOp?
		return false;
	}
}
