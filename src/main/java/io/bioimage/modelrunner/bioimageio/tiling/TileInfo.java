package io.bioimage.modelrunner.bioimageio.tiling;

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
	
	public static TileInfo build(String tensorName, long[] imDims, String imAxesOrder, long[] proposedTileDims, String tileAxesOrder) {
		return new TileInfo(tensorName, imDims, imAxesOrder, proposedTileDims, tileAxesOrder);
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
	public long[] getImageDims() {
		return imDims;
	}

	/**
	 * @return the proposedTileDims
	 */
	public long[] getTileDims() {
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
