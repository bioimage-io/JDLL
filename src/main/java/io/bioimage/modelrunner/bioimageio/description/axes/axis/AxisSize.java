package io.bioimage.modelrunner.bioimageio.description.axes.axis;

import java.util.Map;

public class AxisSize {

	private int min = 1;
	private int step = 0;
	private double offset = 0;
	private String axisID;
	private String ref;
	
	
	
	protected AxisSize(Object object) {
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
