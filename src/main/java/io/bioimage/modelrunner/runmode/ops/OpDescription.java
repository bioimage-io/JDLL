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
import java.util.List;

public class OpDescription {
	
	private boolean isDefault = false;
	
	private String opDir;
	
	private static final List<String> defaultOps = null;
	
	public static OpDescription setupOP(String projectRepo, String opFileName, String opMethodName) {
		
	}
	
	public void defineCondaEnv() {
		
	}
	
	protected void installOp() {
		
	}
	
	public void setInputsInOrder(List<Object> orderedInputList) {
		
	}
	
	public void setNumberOfOuptuts(int nOutputs) {
		
	}

	public String getCondaEnv() {
		return "";
	}
	
	public String getMethodName() {
		return "";
	}
	
	public LinkedHashMap<String, Object> getMethodExtraArgs() {
		return null;
	}
	
	public Object[] getOutputs() {
		return null;
	}
	
	public String getOpDir() {
		return opDir;
	}
}
