package io.bioimage.modelrunner.tiling;

import java.util.List;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.TileFactory;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;

public class TileCalculator {
	
	private final List<TileInfo> tileInfoList;
	
	private final ModelDescriptor descriptor;
	
	private final TileFactory factory;
	
	private TileCalculator(ModelDescriptor descriptor, List<TileInfo> tileInfoList) {
		this.descriptor = descriptor;
		this.tileInfoList = tileInfoList;
		this.factory = TileFactory.init(descriptor);
	}
	
	public static TileCalculator build(ModelDescriptor descriptor, List<TileInfo> tileInfoList) {
		return new TileCalculator(descriptor, tileInfoList);
	}
	
	public boolean validateTileSize() {
		// TODO add code
		return false;
	}
	
	private void calculate() {
		this.validateTileSize();
	}
	
	public void getTileList() {
		
	}
	
	public void getInsertionPoints(String tensorName, int nTile, String axesOrder) {
		
	}
    
    /**
     * 
     * @return name of the input tensors
     */
    public List<String> getInputTensorNames() {
    	return descriptor.getInputTensors().stream()
    			.map(tt -> tt.getTensorID()).collect(Collectors.toList());
    }
    
    /**
     * 
     * @return name of the input tensors
     */
    public List<String> getOutputTensorNames() {
    	return descriptor.getOutputTensors().stream()
    			.map(tt -> tt.getTensorID()).collect(Collectors.toList());
    }
    
    /**
     * 
     * @return size of the tile that is going to be used to process the image
     */
    public long[] getTileSize(String tensorId) {
    	return this.tileSize;
    }
    
    /**
     * 
     * @return size of the roi of each of the tiles that is going to be used to process the image
     */
    public int[] getRoiSize(String tensorId) {
    	return this.roiSize;
    }
    
    /**
     * 
     * @return the position of the closest corner to the center for each of the tiles  with 
     * respect to the original image of the tensor.
     * The positions might be negative as the image that is going to be processed might have padding on the edges
     */
    public List<long[]> getTilePostionsInputImage(String tensorId) {
    	return this.tilePostionsInImage;
    }
    
    /**
     * 
     * @return the position of the closest corner to the center for each of the tiles  with 
     * respect to the original image of the tensor.
     * The positions might be negative as the image that is going to be processed might have padding on the edges
     */
    public List<long[]> getTilePostionsOutputImage(String tensorId) {
    	return this.tilePostionsInImage;
    }
	
	
	

}
