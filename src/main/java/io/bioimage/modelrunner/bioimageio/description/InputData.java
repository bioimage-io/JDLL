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
package io.bioimage.modelrunner.bioimageio.description;

import java.util.Map;

public class InputData {

	private int min = 1;
	private int step = 1;
	private double offset = 0;
	private String axisID;
	private String ref;
	
	
	
	protected InputData(Object object) {
		if (!(object instanceof Map))
			return;
		Map<String, Object> map = (Map<String, Object>) object;
		
		if (map.get("size") == null || !(map.get("size") instanceof Map))
			return;
		
		Map<String, Object> size = (Map<String, Object>) map.get("size");
		if (size.get("min") != null && (size.get("min") instanceof Integer)) 
			min = (int) size.get("min");
		if (size.get("step") != null && (size.get("step") instanceof Integer)) 
			step = (int) size.get("step");
		if (size.get("axis_id") != null && (size.get("axis_id") instanceof String)) 
			axisID = (String) size.get("axis_id");
		if (size.get("tensor_id") != null && (size.get("tensor_id") instanceof String)) 
			ref = (String) size.get("tensor_id");
	}
	
	public int getMin() {
		return this.min;
	}
	
	public int getStep() {
		return this.step;
	}
	
	public double getOffset() {
		return this.offset;
	}
	
	public String getReferenceAxis() {
		return this.axisID;
	}
	
	public String getReferenceTensor() {
		return this.ref;
	}

}
