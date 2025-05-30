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

public class DLModelPytorch extends DLModelPytorchProtected {
	
	protected DLModelPytorch(String modelFile, String callable, String importModule, String weightsPath, Map<String, Object> kwargs)
			throws IOException {
		super(modelFile, callable, importModule, weightsPath, kwargs);
	}

	public static DLModelPytorch create(String modelClass, String callable, String importModule, 
			String modelPath) throws IOException {
		return create(modelClass, callable, importModule, modelPath, new HashMap<String, Object>());
	}
	
	public static DLModelPytorch create(String modelClass, String callable, String importModule, 
			String modelPath, Map<String, Object> kwargs) throws IOException {
		return new DLModelPytorch(modelClass, callable, importModule, modelPath, kwargs);
	}
}
