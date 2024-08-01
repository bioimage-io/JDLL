package io.bioimage.modelrunner.bioimageio.description.axes.axis;

import java.util.List;

public interface Axis {
	
	public String getAxis();
	
	public int getMin();
	
	public int getStep();
	
	public double getScale();

	/**
	 * @return the channelNames
	 */
	public List<String> getChannelNames();

	/**
	 * @return the description
	 */
	public String getDescription();

	/**
	 * @return the concat
	 */
	public boolean isConcat();
	
	public int getHalo();
	
	public double getOffset();
	
	public String getReferenceTensor();
	
	public String getReferenceAxis();

}
