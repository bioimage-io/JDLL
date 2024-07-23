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
	private AxisSize size;
	private int min = 1;
	private int step = 1;
	private int halo = 0;
	
	
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
		if (map.get("halo") != null) 
			this.halo = (int) map.get("halo");
		
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
	
	public String getReferenceTensor() {
		return this.size.getReferenceTensor();
	}
	
	public String getReferenceAxis() {
		return this.size.getReferenceAxis();
	}

}
