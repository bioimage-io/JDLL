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

import java.util.List;
import java.util.Map;

public class AxisV05 implements Axis{

	private String id;
	private String type;
	private String description = "";
	private List<String> channelNames;
	private final String abreviation;
	private boolean concat = false;
	private double scale = 1.0;
	private AxisSize size;
	protected int halo = 0;
	
	private Map<String, Object> originalDescription; 
	
	
	protected AxisV05(Map<String, Object> map) {
		originalDescription = map;
		this.id = (String) map.get("id");
		this.type = (String) map.get("type");
		this.channelNames = (List<String>) map.get("channel_names");
		
		if (map.get("description") != null) 
			this.description = (String) map.get("description");
		if (map.get("concatenable") != null) 
			this.concat = (boolean) map.get("concatenable");
		if (map.get("scale") != null) 
			this.scale = ((Number) map.get("scale")).doubleValue();
		if (map.get("halo") != null) 
			this.halo = ((Number) map.get("halo")).intValue();
		
		this.size = new AxisSize(map.get("size"));
		
		if (this.id == null && this.type == null)
			throw new IllegalArgumentException("Invalid axis configuration: "
					+ "Either 'type' or 'id' must be defined for each axis. "
					+ "Current axis definition is missing both.");
		else if (this.id == null && this.type.equals("space"))
			throw new IllegalArgumentException(String.format(
				    "Invalid axis configuration: When axis type is 'spaces', an 'id' must be defined. " +
				    "Current configuration: type='%s', id=%s", 
				    type, "null"
				));
		else if ((this.id == null || this.id != "b") && this.type.equals("batch"))
			this.abreviation = "b";
		else if (this.id != null && id.equals("channel")) {
			this.abreviation = "c";
			this.size.min = channelNames.size();
		} else if (this.type != null && this.type.equals("channel")) {
			this.abreviation = "c";
			this.size.min = channelNames.size();
		} else
			this.abreviation = this.id;
	}
	
	public String getAxis() {
		return this.abreviation;
	}
	
	public int getMin() {
		if (this.abreviation .equals("c"))
			return this.channelNames != null ? this.channelNames.size() : 1;
		return this.size.getMin();
	}
	
	public int getStep() {
		return this.size.getStep();
	}
	
	public double getScale() {
		return this.scale;
	}

	/**
	 * @return the channelNames
	 */
	public List<String> getChannelNames() {
		return channelNames;
	}

	/**
	 * @return the description
	 */
	public String getDescription() {
		return description;
	}

	/**
	 * @return the concat
	 */
	public boolean isConcat() {
		return concat;
	}
	
	public int getHalo() {
		return this.halo;
	}
	
	public double getOffset() {
		return this.size.getOffset();
	}
	
	public String getReferenceTensor() {
		return this.size.getReferenceTensor();
	}
	
	public String getReferenceAxis() {
		return this.size.getReferenceAxis();
	}
	
	/**
	 * 
	 * @return a map containing the original description used in the Bioimage.io rdf.yaml file
	 */
	public Map<String, Object> getOriginalDescription(){
		return this.originalDescription;
	}

}
