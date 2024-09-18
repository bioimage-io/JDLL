package io.bioimage.modelrunner.tiling;

import java.util.List;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.TileFactory;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;

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
		checkAllTensorsDefined();
		for (TileInfo tile : tileInfoList) {
			String id = tile.getName();
			TensorSpec tensor = descriptor.getInputTensors().stream()
					.filter(tt -> tt.getTensorID().equals(id)).findFirst().orElse(null);
			if (tensor == null)
				throw new IllegalArgumentException("Invalid tiling information: The input tensor named '" 
			            + id + "' does not exist in the model. Please check the model's input tensors "
	            		+ "and provide tiling information for an existing tensor.");
			checkTileDims(tensor, tile);
		}
		checkTilesCombine();
		return false;
	}
	
	private void checkTilesCombine() {
		
	}
	
	private void checkAllTensorsDefined() {
		for (TensorSpec tensor : this.descriptor.getInputTensors()) {
			TileInfo info = tileInfoList.stream()
					.filter(tt -> tt.getName().equals(tensor.getTensorID())).findFirst().orElse(null);
			if (info == null) {
				throw new IllegalArgumentException("Tiling info for input tensor '" + tensor.getTensorID()
													+ "' not defined.");
			}
		}
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
    	return null;
    }
    
    /**
     * 
     * @return size of the roi of each of the tiles that is going to be used to process the image
     */
    public int[] getRoiSize(String tensorId) {
    	return null;
    }
    
    /**
     * 
     * @return the position of the closest corner to the center for each of the tiles  with 
     * respect to the original image of the tensor.
     * The positions might be negative as the image that is going to be processed might have padding on the edges
     */
    public List<long[]> getTilePostionsInputImage(String tensorId) {
    	return null;
    }
    
    /**
     * 
     * @return the position of the closest corner to the center for each of the tiles  with 
     * respect to the original image of the tensor.
     * The positions might be negative as the image that is going to be processed might have padding on the edges
     */
    public List<long[]> getTilePostionsOutputImage(String tensorId) {
    	return null;
    }
	
	private static void checkTileDims(TensorSpec tensor, TileInfo tile) {
		
	}
	
	
	

}
