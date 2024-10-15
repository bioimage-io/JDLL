package io.bioimage.modelrunner.bioimageio.description;

import java.util.List;

public class AxisV04 implements Axis {

	private String description = "";
	private final String abreviation;
	private boolean concat = false;
	private double scale = 1.0;
	private int min = 1;
	private int step = 1;
	protected int halo = 0;
	private double offset = 0;
	String referenceTensor;
	String referenceAxis;
	
	
	protected AxisV04(String abreviation, int min, int step, int halo, double offset, double scale, String ref) {
		this.abreviation = abreviation;
		this.halo = halo;
		this.min = min;
		this.offset = offset;
		this.scale = scale;
		this.step = step;
		this.referenceAxis = abreviation;
		this.referenceTensor = ref;
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
		return null;
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
		return this.offset;
	}
	
	public String getReferenceTensor() {
		return this.referenceTensor;
	}
	
	public String getReferenceAxis() {
		return this.referenceAxis;
	}

}
