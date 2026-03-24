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

import java.util.LinkedHashMap;
import java.util.Map;

public class ModelArchitecture {
	
	private String callable;
	
	private String importModule;
	
	private String source;
	
	private LinkedHashMap<String, Object> kwargs;
	
	/**
	 * Creates a new ModelArchitecture.
	 *
	 * @param map the map parameter.
	 */
	protected ModelArchitecture(Map<String, Object> map) {
		kwargs = new LinkedHashMap<String, Object>();
		if (map.get("callable") != null && map.get("callable") instanceof String)
			callable = (String) map.get("callable");
		if (map.get("import_from") != null && map.get("import_from") instanceof String)
			importModule = (String) map.get("import_from");
		if (map.get("source") != null && map.get("source") instanceof String)
			source = (String) map.get("source");
		if (map.get("kwargs") != null && map.get("kwargs") instanceof Map) {
			for (String kk : ((Map<String, Object>) map.get("kwargs")).keySet())
				kwargs.put(kk, ((Map<String, Object>) map.get("kwargs")).get(kk));
		}
	}
	
	/**
	 * Gets callable.
	 *
	 * @return the resulting string.
	 */
	public String getCallable() {
		return callable;
	}
	
	/**
	 * Gets import module.
	 *
	 * @return the resulting string.
	 */
	public String getImportModule() {
		return importModule;
	}
	
	/**
	 * Gets source.
	 *
	 * @return the resulting string.
	 */
	public String getSource() {
		return source;
	}
	
	/**
	 * Gets kwargs.
	 *
	 * @return the resulting value.
	 */
	public Map<String, Object> getKwargs() {
		return kwargs;
	}

}
