package io.bioimage.modelrunner.tiling;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.bioimage.modelrunner.bioimageio.TileFactory;
import io.bioimage.modelrunner.bioimageio.description.Axis;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.Constants;
import net.imglib2.RandomAccessibleInterval;

public class TileCalculator {
	
	private final List<TileInfo> inputTileInfo;
	
	private List<TileInfo> outputTileInfo;
	
	private final ModelDescriptor descriptor;
	
	private final TileFactory factory;
	
	private final LinkedHashMap<String, PatchSpec> input = new LinkedHashMap<String, PatchSpec>();
	
	private final LinkedHashMap<String, PatchSpec> output = new LinkedHashMap<String, PatchSpec>();
	
	private TileCalculator(ModelDescriptor descriptor, List<TileInfo> tileInfoList) {
		this.descriptor = descriptor;
		this.inputTileInfo = tileInfoList;
		validate();
		this.factory = TileFactory.init(descriptor);
	}
	
	public static TileCalculator build(ModelDescriptor descriptor, List<TileInfo> tileInfoList) {
		return new TileCalculator(descriptor, tileInfoList);
	}
	
	private void validate() {
		checkAllTensorsDefined();
		validateTileVsImageSize();
		validateStepMin();
		getOutputTiles();
		validateTileVsHalo();
		validateTileVsImageChannel();
		checkTilesCombine();
	}
	
	private void getOutputTiles() {
		outputTileInfo = new ArrayList<TileInfo>();
		for (TensorSpec tt : this.descriptor.getOutputTensors()) {
			String outAxesOrder = tt.getAxesOrder();
			long[] tileSize = new long[outAxesOrder.length()];
			long[] imagSize = new long[outAxesOrder.length()];
			int i = -1;
			for (Axis ax : tt.getAxesInfo().getAxesList()) {
				i ++;
				if (ax.getStep() == 0 && !descriptor.isTilingAllowed()) {
					tileSize[i] = ax.getMin();
					imagSize[i] = ax.getMin();
				} else if (ax.getStep() != 0 && !descriptor.isTilingAllowed()) {
					throw new IllegalArgumentException(""
							+ "Model specs too complex for JDLL. "
							+ "Please contact the team and create and issue attaching the rdf.yaml file"
							+ " so we can troubleshoot at: " + Constants.ISSUES_LINK);
				} else if (ax.getStep() == 0 && ax.getReferenceTensor() == null) {
					TensorSpec intt = descriptor.getInputTensors().stream()
							.filter(t -> t.isImage()).findFirst().orElse(null);
					TileInfo inTile = inputTileInfo.stream()
							.filter(t -> t.getName().equals(intt.getTensorID())).findFirst().orElse(null);
					int indTile = inTile.getTileAxesOrder().indexOf(ax.getAxis());
					int indIm = inTile.getImageAxesOrder().indexOf(ax.getAxis());
					if (indTile == -1 || indIm == -1)
						throw new IllegalArgumentException(""
								+ "Model specs too complex for JDLL. "
								+ "Please contact the team and create and issue attaching the rdf.yaml file"
								+ " so we can troubleshoot at: " + Constants.ISSUES_LINK);
					double factor = (double) inTile.getImageDimensions()[indIm] / inTile.getProposedTileDimensions()[indTile];
					if (Math.floor(ax.getMin() * factor) != ax.getMin() * factor)
						throw new IllegalArgumentException(""
								+ "Model specs too complex for JDLL. "
								+ "Please contact the team and create and issue attaching the rdf.yaml file"
								+ " so we can troubleshoot at: " + Constants.ISSUES_LINK);
					imagSize[i] = (long) (ax.getMin() * factor);
					tileSize[i] = (long) (ax.getMin() * factor);
				} else if (ax.getStep() == 0) {
					TileInfo inTile = inputTileInfo.stream()
							.filter(t -> t.getName().equals(ax.getReferenceTensor())).findFirst().orElse(null);
					int indTile = inTile.getTileAxesOrder().indexOf(ax.getReferenceAxis());
					int indIm = inTile.getImageAxesOrder().indexOf(ax.getReferenceAxis());
					if (indTile == -1 || indIm == -1)
						throw new IllegalArgumentException(""
								+ "Model specs too complex for JDLL. "
								+ "Please contact the team and create and issue attaching the rdf.yaml file"
								+ " so we can troubleshoot at: " + Constants.ISSUES_LINK);
					double factor = (double) inTile.getImageDimensions()[indIm] / inTile.getProposedTileDimensions()[indTile];
					if (Math.floor(ax.getMin() * factor) != ax.getMin() * factor)
						throw new IllegalArgumentException(""
								+ "Model specs too complex for JDLL. "
								+ "Please contact the team and create and issue attaching the rdf.yaml file"
								+ " so we can troubleshoot at: " + Constants.ISSUES_LINK);
					imagSize[i] = (long) (ax.getMin() * factor);
					tileSize[i] = (long) (ax.getMin() * factor);
				} else if (ax.getReferenceTensor() == null) {
					throw new IllegalArgumentException(""
							+ "Model specs too complex for JDLL. "
							+ "Please contact the team and create and issue attaching the rdf.yaml file"
							+ " so we can troubleshoot at: " + Constants.ISSUES_LINK);
				} else if (ax.getStep() != 0) {
					TileInfo inTile = inputTileInfo.stream()
							.filter(t -> t.getName().equals(ax.getReferenceTensor())).findFirst().orElse(null);
					int indTile = inTile.getTileAxesOrder().indexOf(ax.getReferenceAxis());
					int indIm = inTile.getImageAxesOrder().indexOf(ax.getReferenceAxis());
					imagSize[i] = (long) (inTile.getImageDimensions()[indIm] * ax.getScale());
					tileSize[i] = (long) (inTile.getProposedTileDimensions()[indTile] * ax.getScale() + ax.getOffset() * 2);
				} else {
					throw new IllegalArgumentException(""
							+ "Model specs too complex for JDLL. "
							+ "Please contact the team and create and issue attaching the rdf.yaml file"
							+ " so we can troubleshoot at: " + Constants.ISSUES_LINK);
				}
				outputTileInfo.add(TileInfo.build(tt.getTensorID(), imagSize, outAxesOrder, tileSize, outAxesOrder));
			}
		}
	}
	
	private void validateTileVsHalo() {
		for (TensorSpec tt : this.descriptor.getOutputTensors()) {
			for (Axis ax : tt.getAxesInfo().getAxesList()) {
				String ref = ax.getReferenceTensor();
				if (ref == null) continue;
				if (ax.getMin() != 0 && ax.getStep() == 0)
					continue;
				TileInfo tile = this.inputTileInfo.stream().filter(til -> til.getName().equals(ref)).findFirst().orElse(null);
				if (tile == null) throw new IllegalArgumentException("Tile specs of input tensor '" + ref + "' not defined.");
				String axisStr = ax.getReferenceAxis();
				String tileAxes = tile.getTileAxesOrder();
				int ind = tileAxes.indexOf(axisStr);
				long refSize = 1;
				if (ind != -1)
					refSize = tile.getProposedTileDimensions()[ind];
				double outSize = ax.getScale() * refSize + ax.getOffset() * 2;
				if (outSize - ax.getHalo() * 2 <= 0)
					throw new IllegalArgumentException("Input size too small, halo would be bigger than "
							+ "the image accross dimension '" + axisStr + "'. Toal halo = " + ax.getHalo() * 2
							+ ", image size = " + outSize + ".");
			}
		}
		
	}
	
	private void validateStepMin() {
		for (TileInfo tile : this.inputTileInfo) {
			TensorSpec tt = this.descriptor.findInputTensor(tile.getName());
			if (tt == null) continue;
    		String axesTile = tile.getTileAxesOrder();
    		long[] tileDims = tile.getProposedTileDimensions();
    		String axesTensor = tt.getAxesOrder();
    		axesTile = addMissingAxes(axesTensor, axesTile);
    		axesTensor = addMissingAxes(axesTile, axesTensor);
    		tileDims = arrayToWantedAxesOrderAddOnes(tileDims, tile.getTileAxesOrder(), axesTile);
    		int[] min = arrayToWantedAxesOrderAddOnes(tt.getMinTileSizeArr(), tt.getAxesOrder(), axesTile);
    		int[] step = arrayToWantedAxesOrderAddZeros(tt.getTileStepArr(), tt.getAxesOrder(), axesTile);
    		
    		for (int i = 0; i < tileDims.length; i ++) {
    			if (tileDims[i] != min[i] && step[i] == 0)
    				throw new IllegalArgumentException("Invalid tile size for axis '" + axesTile.split("")[i].toUpperCase()
    						+ "'. Only allowed tile size for this axis is: " + min[i]);
    			else if ((tileDims[i] - min[i]) % step[i] != 0)
    				throw new IllegalArgumentException("Invalid tile size for axis '" + axesTile.split("")[i].toUpperCase()
    						+ "'. Tile size for this axis should satisfy: " + min[i] + " + n x " + step[i]
    						+ " where n can be any positive integer.");
    		}
		}
	}
	
	private void validateTileVsImageChannel() {
    	for (TileInfo tile : this.inputTileInfo) {
    		String tileAxes = tile.getTileAxesOrder();
    		String imageAxes = tile.getImageAxesOrder();
    		long[] tileSize = tile.getProposedTileDimensions();
    		long[] imSize = tile.getImageDimensions();
    		int indTile = tileAxes.indexOf("c");
    		int indIm = imageAxes.indexOf("c");
    		if (indIm != -1 && indTile != -1 && tileSize[indTile] != imSize[indIm])
    			throw new IllegalArgumentException("Tiling cannot happen accross the channel dimension. "
    					+ "The tile number of channels (" + tileSize[indTile] + ") must be the same "
						+ "as the image number of channels (" + imSize[indIm] + ").");
    		else if (indIm == -1 && tileSize[indTile] != 1)
    			throw new IllegalArgumentException("Tiling cannot happen accross the channel dimension. "
    					+ "The tile number of channels (" + tileSize[indTile] + ") must be the same "
						+ "as the image number of channels (" + 1 + ").");
    		else if (indTile == -1 && imSize[indIm] != 1)
    			throw new IllegalArgumentException("Tiling cannot happen accross the channel dimension. "
    					+ "The tile number of channels (" + 1 + ") must be the same "
						+ "as the image number of channels (" + imSize[indIm] + ").");
    	}
	}
	
    private void validateTileVsImageSize() throws IllegalArgumentException {
    	for (TileInfo tile : this.inputTileInfo) {
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
			TileInfo info = inputTileInfo.stream()
					.filter(tt -> tt.getName().equals(tensor.getTensorID())).findFirst().orElse(null);
			if (info == null) {
				throw new IllegalArgumentException("Tiling info for input tensor '" + tensor.getTensorID()
													+ "' not defined.");
			}
		}
	}
	
	private void calculate() {
		for (TensorSpec tt : this.descriptor.getInputTensors()) {
			TileInfo tile = inputTileInfo.stream()
					.filter(til -> til.getName().equals(tt.getTensorID())).findFirst().orElse(null);
			input.put(tt.getTensorID(), computePatchSpecs(tt, tile));
		}
	}

    /**
     * Compute the patch details needed to perform the tiling strategy. The calculations
     * obtain the input patch, the padding needed at each side and the number of patches
     * needed for every tensor.
     * 
     * @param spec
     * 	specs of the tensor
     * @param rai
     * 	ImgLib2 rai, backend of a tensor, that is going to be tiled
     * @param tileSize
     * 	the size of the tile selected to process the image
     * 
     * @return an object containing the specs needed to perform patching for the particular tensor
     */
    private PatchSpec computePatchSpecs(TensorSpec spec, TileInfo tile)
    {
    	long[] imSize = arrayToWantedAxesOrderAddOnes(tile.getImageDimensions(), 
    			tile.getImageAxesOrder(), spec.getAxesInfo().getAxesOrder());
    	long[] tileSize = arrayToWantedAxesOrderAddOnes(tile.getProposedTileDimensions(), 
    			tile.getTileAxesOrder(), spec.getAxesInfo().getAxesOrder());
        int[][] paddingSize = new int[2][tileSize.length];
        // REgard that the input halo represents the output halo + offset 
        // and must be divisible by 0.5. 
        int[] halo = spec.getHaloArr();
        if (!descriptor.isPyramidal() && this.descriptor.isTilingAllowed()) {
        	// In the case that padding is asymmetrical, the left upper padding has the extra pixel
            for (int i = 0; i < halo.length; i ++) {paddingSize[0][i] = (int) Math.ceil(halo[i]);}
            // In the case that padding is asymmetrical, the right bottom padding has one pixel less
            for (int i = 0; i < halo.length; i ++) {paddingSize[1][i] = (int) Math.floor(halo[i]);}
            
        }
        int[] patchGridSize = new int[imSize.length];
        for (int i = 0; i < patchGridSize.length; i ++) patchGridSize[i] = 1;
        if (descriptor.isTilingAllowed()) {
            patchGridSize = IntStream.range(0, tileSize.length)
                    .map(i -> (int) Math.ceil((double) imSize[i] / ((double) tileSize[i] - halo[i] * 2)))
                    .toArray();
        }
        // For the cases when the patch is bigger than the  image size, share the
        // padding between both sides of the image
        paddingSize[0] = IntStream.range(0, tileSize.length)
                .map(i -> 
                	(int) Math.max(paddingSize[0][i],
                			Math.ceil( (double) (tileSize[i] - imSize[i]) / 2))
                ).toArray();
        paddingSize[1] = IntStream.range(0, tileSize.length)
            .map(i -> (int) Math.max( paddingSize[1][i], 
            		tileSize[i] - imSize[i] - paddingSize[0][i])).toArray();

        return PatchSpec.create(spec.getTensorID(), tileSize, patchGridSize, paddingSize, imSize);
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
    public static int[] arrayToWantedAxesOrderAddOnes(int[] size, String orginalAxes, String targetAxes) {
    	orginalAxes = orginalAxes.toLowerCase();
    	String[] axesArr = targetAxes.toLowerCase().split("");
    	int[] finalSize = new int[targetAxes.length()];
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
    public static long[] arrayToWantedAxesOrderAddZeros(long[] size, String orginalAxes, String targetAxes) {
    	orginalAxes = orginalAxes.toLowerCase();
    	String[] axesArr = targetAxes.toLowerCase().split("");
    	long[] finalSize = new long[targetAxes.length()];
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
    
    /**
	 * Compare two axes order strings and adds the dimensions of axes1 that are not present in 
	 * axes2 to axes2. Special rules are applied for axes 'b' and 't':
	 * - If 'b' is present in axes1 but 't' is in axes2, 'b' is skipped.
	 * - If 't' is present in axes1 but 'b' is in axes2, 't' is added to axes2.
	 * 
	 * For example:
	 * <pre>
	 *     String result1 = addMissingAxes("xyz", "xz");
	 *     // result1 will be "xyz" since 'y' is added to "xz"
	 *     
	 *     String result2 = addMissingAxes("xyz", "xyc");
	 *     // result2 will be "xycz" 
	 * </pre>
	 * 
	 * @param axes1 The source axes order string from which missing axes are added.
	 * @param axes2 The target axes order string where missing axes are added.
	 * @return The modified axes2 string including missing axes from axes1.
     */
    public static String addMissingAxes(String axes1, String axes2) {
    	for (String ax : axes1.split("")) {
    		if (ax.equals("b") && axes2.indexOf(ax) == -1 && axes2.indexOf("t") != -1)
    			continue;
			else if (ax.equals("t") && axes2.indexOf(ax) == -1 && axes2.indexOf("b") != -1)
    			axes2 += ax;
			else if (axes2.indexOf(ax) == -1)
    			axes2 += ax;
    	}
    	return axes2;
    }
	
	
	

}
