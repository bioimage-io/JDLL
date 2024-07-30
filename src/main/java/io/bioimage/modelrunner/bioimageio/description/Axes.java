package io.bioimage.modelrunner.bioimageio.description;

import java.util.List;

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
