package io.bioimage.modelrunner.bioimageio.description.axes;

import java.util.List;

import io.bioimage.modelrunner.bioimageio.description.axes.axis.Axis;

public interface Axes {
	
	public String getAxesOrder();

	/**
	 * @return the axesList
	 */
	public List<Axis> getAxesList();

	public int[] getMinTileSizeArr();

	public int[] getTileStepArr();

	public double[] getTileScaleArr();
	
	public Axis getAxis(String abreviation);

}
