package io.bioimage.modelrunner.tiling;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
		validateTileSize();
		this.factory = TileFactory.init(descriptor);
	}
	
	public static TileCalculator build(ModelDescriptor descriptor, List<TileInfo> tileInfoList) {
		return new TileCalculator(descriptor, tileInfoList);
	}
	
	private boolean validateTileSize() {
		checkAllTensorsDefined();
		checkTileDims(tensor, tile);
		validateTileVsImageSize();
		checkTilesCombine();
		return false;
	}
	
	private void validate 
	
	private void validateStepMin() {
		
	}
	
    private void validateTileVsImageSize() throws IllegalArgumentException {
    	for (TileInfo tile : this.tileInfoList) {
    		String axesTile = tile.getTileAxesOrder();
    		String axesImage = tile.getImageAxesOrder();
    		long[] tileDims = tile.getProposedTileDimensions();
    		checkAxisSize(tile);
    		long[] imDims = arrayToWantedAxesOrderAddOnes(tile.getImageDimensions(), axesImage, axesTile);
    		for (int i = 0; i < axesTile.length(); i ++) {
    			int indIm = axesImage.indexOf(axesTile.split("")[i]);
    			if (imDims[indIm] * 3 < tileDims[i])
    				throw new IllegalArgumentException("Error in the axes size selected. "
    						+ "The axes size introduced in any of the dimensions cannot "
    						+ "be bigger than 3 times the image size of that same axes. "
    						+ "The image selected has " + axesTile.split("")[i] + "-dimension of size "
							+ imDims[indIm] + "and the tile is of size " + tileDims[i] + "."
							+ " Maxmum tile size for " + axesTile.split("")[i] + "-axis in this image is "
							+ imDims[indIm] * 3);
    		}
    	}
    }
    
    private static void checkAxisSize(TileInfo tile) {
		String axesTile = tile.getTileAxesOrder();
		long[] tileDims = tile.getProposedTileDimensions();
		if (axesTile.length() != tileDims.length)
			throw new IllegalArgumentException("The tile dimensions and tile axes should be of the same length:"
					+ " " + axesTile + " (" + axesTile.length() + ") vs " + Arrays.toString(tileDims) 
					+ " (" + tileDims.length + ")");
		String axesImage = tile.getImageAxesOrder();
		long[] imDims = tile.getImageDimensions();
		if (axesImage.length() != imDims.length)
			throw new IllegalArgumentException("The image dimensions and image axes should be of the same length:"
					+ " " + axesImage + " (" + axesImage.length() + ") vs " + Arrays.toString(imDims) 
					+ " (" + imDims.length + ")");
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
    
    /**
     * Convert the array following given axes order into
     *  another int[] which follows the target axes order
     *  The newly added components will be ones.
     * @param size
     * 	original array following the original axes order
     * @param orginalAxes
     * 	axes order of the original array
     * @param targetAxes
     * 	axes order of the target array
     * @return a size array in the order of the tensor of interest
     */
    public static long[] arrayToWantedAxesOrderAddOnes(long[] size, String orginalAxes, String targetAxes) {
    	orginalAxes = orginalAxes.toLowerCase();
    	String[] axesArr = targetAxes.toLowerCase().split("");
    	long[] finalSize = new long[targetAxes.length()];
    	for (int i = 0; i < finalSize.length; i ++) {
    		int ind = orginalAxes.indexOf(axesArr[i]);
    		if (ind == -1) {
    			finalSize[i] = 1;
    		} else {
    			finalSize[i] = size[ind];
    		}
    	}
    	return finalSize;
    }
    
    /**
     * Convert the array following given axes order into
     *  another float[] which follows the target axes order
     *  The newly added components will be ones.
     * @param size
     * 	original array following the original axes order
     * @param orginalAxes
     * 	axes order of the original array
     * @param targetAxes
     * 	axes order of the target array
     * @return a size array in the order of the tensor of interest
     */
    public static float[] arrayToWantedAxesOrderAddOnes(float[] size, String orginalAxes, String targetAxes) {
    	orginalAxes = orginalAxes.toLowerCase();
    	String[] axesArr = targetAxes.toLowerCase().split("");
    	float[] finalSize = new float[targetAxes.length()];
    	for (int i = 0; i < finalSize.length; i ++) {
    		int ind = orginalAxes.indexOf(axesArr[i]);
    		if (ind == -1) {
    			finalSize[i] = 1;
    		} else {
    			finalSize[i] = size[ind];
    		}
    	}
    	return finalSize;
    }
    
    /**
     * Convert the array following given axes order into
     *  another float[] which follows the target axes order.
     *  The newly added components will be zeros.
     * @param size
     * 	original array following the original axes order
     * @param orginalAxes
     * 	axes order of the original array
     * @param targetAxes
     * 	axes order of the target array
     * @return a size array in the order of the tensor of interest
     */
    public static float[] arrayToWantedAxesOrderAddZeros(float[] size, String orginalAxes, String targetAxes) {
    	orginalAxes = orginalAxes.toLowerCase();
    	String[] axesArr = targetAxes.toLowerCase().split("");
    	float[] finalSize = new float[targetAxes.length()];
    	for (int i = 0; i < finalSize.length; i ++) {
    		int ind = orginalAxes.indexOf(axesArr[i]);
    		if (ind == -1) 
    			continue;
    		finalSize[i] = size[ind];
    	}
    	return finalSize;
    }
    
    /**
     * Convert the array following given axes order into
     *  another int[] which follows the target axes order.
     *  The newly added components will be zeros.
     * @param size
     * 	original array following the original axes order
     * @param orginalAxes
     * 	axes order of the original array
     * @param targetAxes
     * 	axes order of the target array
     * @return a size array in the order of the tensor of interest
     */
    public static int[] arrayToWantedAxesOrderAddZeros(int[] size, String orginalAxes, String targetAxes) {
    	orginalAxes = orginalAxes.toLowerCase();
    	String[] axesArr = targetAxes.toLowerCase().split("");
    	int[] finalSize = new int[targetAxes.length()];
    	for (int i = 0; i < finalSize.length; i ++) {
    		int ind = orginalAxes.indexOf(axesArr[i]);
    		if (ind == -1) 
    			continue;
    		finalSize[i] = size[ind];
    	}
    	return finalSize;
    }
	
	
	

}
