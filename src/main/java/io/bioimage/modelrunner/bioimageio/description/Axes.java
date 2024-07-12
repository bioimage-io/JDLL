package io.bioimage.modelrunner.bioimageio.description;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Axes {
	
	private final List<Axis> axesList;
	
	private final String axesOrder;
	
	protected Axes(List<Object> axesList) {
		List<Axis> axesListInit = new ArrayList<Axis>();
		String order = "";
		for (Object axisObject : axesList) {
			if (!(axisObject instanceof Map))
				throw new IllegalArgumentException("The input argument should be a list of maps. "
						+ "Go to the Bioimage.io specs documentation for more info.");
			Axis axis = new Axis((Map<String, Object>) axisObject);
			axesListInit.add(axis);
			order += axis.getAxis();
		}
		this.axesList = axesListInit;
		this.axesOrder = order; 
	}
	
	public String getAxesOrder() {
		return this.axesOrder;
	}

	/**
	 * @return the axesList
	 */
	public List<Axis> getAxesList() {
		return axesList;
	}

}
