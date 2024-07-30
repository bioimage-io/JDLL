package io.bioimage.modelrunner.bioimageio.description;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class AxesV04 implements Axes{
	
	private final List<Axis> axesList;
	
	private final String axesOrder;
	
	private final double[] scaleArr;
	
	private final int[] minArr;
	
	private final int[] stepArr;
	
	protected AxesV04(Map<String, Object> tensorSpecMap) {
		List<Axis> axesListInit = new ArrayList<Axis>();
		String order = "";
		int[] minArr = new int[axesList.size()];
		int[] stepArr = new int[axesList.size()];
		double[] scaleArr = new double[axesList.size()];
		int c = 0;
		for (Object axisObject : axesList) {
			if (!(axisObject instanceof Map))
				throw new IllegalArgumentException("The input argument should be a list of maps. "
						+ "Go to the Bioimage.io specs documentation for more info.");
			Axis axis = new Axis((Map<String, Object>) axisObject);
			axesListInit.add(axis);
			order += axis.getAxis();
			minArr[c] = axis.getMin();
			stepArr[c] = axis.getStep();
			scaleArr[c] = axis.getScale();
		}
		this.axesList = axesListInit;
		this.axesOrder = order;
		this.scaleArr = scaleArr;
		this.minArr = minArr;
		this.stepArr = stepArr; 
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

	public int[] getMinTileSizeArr() {
		return this.minArr;
	}

	public int[] getTileStepArr() {
		return this.stepArr;
	}

	public double[] getTileScaleArr() {
		return this.scaleArr;
	}
	
	public Axis getAxis(String abreviation) {
		return axesList.stream().filter(ax -> ax.getAxis().equals(abreviation)).findFirst().orElse(null);
	}

}
