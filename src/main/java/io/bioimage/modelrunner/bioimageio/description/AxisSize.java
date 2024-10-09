package io.bioimage.modelrunner.bioimageio.description;

import java.util.Map;

public class AxisSize {

	private int min = 1;
	private int step = 0;
	private double offset = 0;
	private String axisID;
	private String ref;
	
	
	
	protected AxisSize(Object object) {
		if (object instanceof Number) {
			min = (int) object;
			step = 0;
		}
			
		if (!(object instanceof Map))
			return;
		Map<String, Object> map = (Map<String, Object>) object;
		
		
		if (map.get("min") != null && (map.get("min") instanceof Number)) 
			min = ((Number) map.get("min")).intValue();
		if (map.get("step") != null && (map.get("step") instanceof Number)) 
			step = ((Number) map.get("step")).intValue();
		if (map.get("offset") != null && (map.get("offset") instanceof Number)) 
			offset = ((Number) map.get("offset")).doubleValue();
		if (map.get("axis_id") != null && (map.get("axis_id") instanceof String)) 
			axisID = (String) map.get("axis_id");
		if (map.get("tensor_id") != null && (map.get("tensor_id") instanceof String)) 
			ref = (String) map.get("tensor_id");
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
