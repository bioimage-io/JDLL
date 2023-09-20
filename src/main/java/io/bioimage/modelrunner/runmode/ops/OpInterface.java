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

import java.util.List;

/**
 * Interface that every OP needs to implement to be able to be run as a RunMode
 * @author Carlos Garcia Lopez de Haro
 *
 */
public interface OpInterface {
	/**
	 * The imports that need to be added to the Python code to run the OP
	 * @return a String containing a Python snippet with the imports needed to use the
	 * OP
	 */
	public String getOpImport();
	/**
	 * Number of outputs that the OP method will produce
	 * @return number of ouptuts that the OP method will produce
	 */
	public int getNumberOfOutputs();
	
	public void installOp();
	/**
	 * Set the inputs to the OP in order
	 * @param orderedInputList
	 * 	the inputs we want to provide to the OP in the same order as they are defined in the OP
	 */
	public void setInputsInOrder(List<Object> orderedInputList);
	/**
	 * Get the Conda env needed to run the OP
	 * 
	 * @return the Conda env required to run the OP
	 */
	public String getCondaEnv();
	/**
	 * The name of the method that is used to call the OP
	 * @return a String with the name of the method used to call the OP
	 */
	public String getMethodName();
	/**
	 * Directory where the OP is stored
	 * @return tirectory where the OP is stored
	 */
	public String getOpDir();
}
