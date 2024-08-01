package io.bioimage.modelrunner.bioimageio;

public class ImageInfo {
	
	private final String name;
	
	private final String axesOrder;
	
	private final long[] dims;
	
	public ImageInfo(String tensorName, String axesOrder, long[] dims) {
		this.dims = dims;
		this.name = tensorName;
		this.axesOrder = axesOrder;
	}
	
	public String getAxesOrder() {
		return this.axesOrder;
	}
	
	public String getTensorName() {
		return this.name;
	}
	
	public long[] getDimensions() {
		return this.dims;
	}

}
