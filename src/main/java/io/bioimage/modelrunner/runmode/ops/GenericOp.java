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

import java.io.File;
import java.util.LinkedHashMap;

/**
 * A generic OP that allows running Python code from JDLL if it is formatted correctly
 * @author Carlos Javier Garcia Lopez de Haro
 *
 */
public class GenericOp implements OpInterface {
	
	private String opFilePath;
	
	private String envPath;
	
	private String methodName;
	
	private int nOutputs;
	
	private LinkedHashMap<String, Object> inputsMap;
	
	
	public static void main(String[] args) {
	}
	
	public static GenericOp create(String envYaml, String script, String method, int nOutputs) {
		GenericOp op = new GenericOp();
		op.envPath = envYaml;
		op.opFilePath = script;
		op.methodName = method;
		op.nOutputs = nOutputs;
		return op;
	}
	
	public void setInputs(LinkedHashMap<String, Object> kwargs) {
		this.inputsMap = kwargs;
	}

	@Override
	public String getOpPythonFilename() {
		return new File(this.opFilePath).getName();
	}

	@Override
	public int getNumberOfOutputs() {
		return nOutputs;
	}

	@Override
	public void installOp() {
		// TODO 
	}

	@Override
	public LinkedHashMap<String, Object> getOpInputs() {
		return this.inputsMap;
	}

	@Override
	public String getCondaEnv() {
		return envPath;
	}

	@Override
	public String getMethodName() {
		return this.methodName;
	}

	@Override
	public String getOpDir() {
		return new File(opFilePath).getParent();
	}

	@Override
	public boolean isOpInstalled() {
		// TODO maybe remove this method? Make the check at installOp?
		return false;
	}
}
