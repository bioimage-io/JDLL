package io.bioimage.modelrunner.tiling;

public class TileInfo {
	
	
	private final String name;
	
	private final long[] imDims;
	
	private final long[] proposedTileDims;
	
	private final String imAxesOrder;
	
	private final String tileAxesOrder;
	
	private TileInfo(String tensorName, long[] imDims, String imAxesOrder, long[] proposedTileDims, String tileAxesOrder) {
		this.name = tensorName;
		this.imAxesOrder = imAxesOrder;
		this.imDims = imDims;
		this.proposedTileDims = proposedTileDims;
		this.tileAxesOrder = tileAxesOrder;
	}

	/**
	 * @return the name
	 */
	public String getName() {
		return name;
	}

	/**
	 * @return the imDims
	 */
	public long[] getImageDimensions() {
		return imDims;
	}

	/**
	 * @return the proposedTileDims
	 */
	public long[] getProposedTileDimensions() {
		return proposedTileDims;
	}

	/**
	 * @return the imAxesOrder
	 */
	public String getImageAxesOrder() {
		return imAxesOrder;
	}

	/**
	 * @return the tileAxesOrder
	 */
	public String getTileAxesOrder() {
		return tileAxesOrder;
	}
	
	

}
