package io.bioimage.modelrunner.bioimageio.description;

import java.util.List;
import java.util.Map;

public class Axis {

	private String id;
	private String type;
	private String description = "";
	private List<String> channelNames;
	private final String abreviation;
	private boolean concat = false;
	private double scale = 1.0;
	private int min = 1;
	private int step = 1;
	
	
	protected Axis(Map<String, Object> map) {
		this.id = (String) map.get("id");
		this.type = (String) map.get("type");
		this.channelNames = (List<String>) map.get("channel_names");
		
		if (map.get("description") != null) 
			this.description = (String) map.get("description");
		if (map.get("concatenable") != null) 
			this.concat = (boolean) map.get("concatenable");
		if (map.get("scale") != null) 
			this.scale = (double) map.get("scale");
		
		
		if (map.get("size") != null && (map.get("size") instanceof Map)) {
			Map<String, Object> size = (Map<String, Object>) map.get("size");
			if (size.get("min") != null && (size.get("min") instanceof Integer)) 
				min = (int) size.get("min");
			if (size.get("step") != null && (size.get("step") instanceof Integer)) 
				step = (int) size.get("step");
		}
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
		else if (this.id == null && this.type.equals("batch"))
			this.abreviation = "b";
		else if (this.id != null && id.equals("channel"))
			this.abreviation = "c";
		else if (this.type != null && this.type.equals("channel"))
			this.abreviation = "c";
		else
			this.abreviation = this.id;
	}
	
	public String getAxis() {
		return this.abreviation;
	}
	
	public int getMin() {
		return this.min;
	}
	
	public int getStep() {
		return this.step;
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

}
