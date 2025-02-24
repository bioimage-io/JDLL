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

package io.bioimage.modelrunner.bioimageio.description.weights;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.utils.YAMLUtils;

public class ModelDependencies {
	
	private String envFile;
	
	private LinkedHashMap<String, Object> kwargs;
	
	protected ModelDependencies(Map<String, Object> map) {
		this.envFile = (String) map.get("source");
	}
	
	public String getSource() {
		return envFile;
	}

	public static List<String> getDependencies(ModelDescriptor descriptor, WeightFormat weights) {
		List<String> deps = new ArrayList<String>();
		if (weights.getEnvDependencies() == null || weights.getEnvDependencies().getSource() == null)
			return deps;
		if (descriptor.getModelPath() != null) {
			String path = descriptor.getModelPath() + File.separator + weights.getEnvDependencies().getSource();
			Map<String, Object> map = YAMLUtils.loadFromString(path);
			if (map.get("dependencies") != null && map.get("dependencies") instanceof String)
				return (List<String>) map.get("dependencies");
		}
		String url = descriptor.getModelURL() + weights.getEnvDependencies().getSource();
		try {
			String stringRDF = BioimageioRepo.getJSONFromUrl(url);
			Map<String,Object> map = YAMLUtils.loadFromString(stringRDF);
			if (map.get("dependencies") != null && map.get("dependencies") instanceof String)
				return (List<String>) map.get("dependencies");
		} catch (InterruptedException e) {
			return deps;
		}
		return deps;
	}

}
