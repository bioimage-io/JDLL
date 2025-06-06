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
/**
 * 
 */
package io.bioimage.modelrunner.model.python;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Class that contains the methods to use a Pytorch model from JDLL.
 * The model should be compatible with the default environment (Biapy environment) or the environmemt
 * preferred.
 * 
 * The callable to instantiate the model should be defined as well as the module from where it has to be imported
 * or the Python file that contains its declaration.
 * 
 */
public class DLModelPytorch extends DLModelPytorchProtected {
	
	protected DLModelPytorch(String modelFile, String callable, String importModule, String weightsPath, Map<String, Object> kwargs)
			throws IOException {
		super(modelFile, callable, importModule, weightsPath, kwargs);
	}

	/**
	 * Create a Pytorch model that can run from Java
	 * @param modelClass
	 * 	file where the class of the model is defined. It the class can be imported this param can be null as long as the 
	 * 	module from which the class can be imported is defined at 'importModule'
	 * @param callable
	 * 	class of the model
	 * @param importModule
	 * 	module from where the model class is imported. It can be null it we provide a file that contains the class
	 * @param modelPath
	 * 	path to the Pytorch weights
	 * @return a Java class that can do inference using a Python Pytorch model
	 * @throws IOException if there is any error trying to run Python from Java
	 */
	public static DLModelPytorch create(String modelClass, String callable, String importModule, 
			String modelPath) throws IOException {
		return create(modelClass, callable, importModule, modelPath, new HashMap<String, Object>());
	}

	/**
	 * Create a Pytorch model that can run from Java
	 * @param modelClass
	 * 	file where the class of the model is defined. It the class can be imported this param can be null as long as the 
	 * 	module from which the class can be imported is defined at 'importModule'
	 * @param callable
	 * 	class of the model
	 * @param importModule
	 * 	module from where the model class is imported. It can be null it we provide a file that contains the class
	 * @param modelPath
	 * 	path to the Pytorch weights
	 * @param kwargs
	 * 	other kwargs that might be needed to instantiate the model
	 * @return a Java class that can do inference using a Python Pytorch model
	 * @throws IOException if there is any error trying to run Python from Java
	 */
	public static DLModelPytorch create(String modelClass, String callable, String importModule, 
			String modelPath, Map<String, Object> kwargs) throws IOException {
		return new DLModelPytorch(modelClass, callable, importModule, modelPath, kwargs);
	}
}
