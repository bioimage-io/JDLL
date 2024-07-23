package io.bioimage.modelrunner.bioimageio.description;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.tiling.PatchGridCalculator;

public class TileFactory {
	
	private final ModelDescriptor descriptor;
    
    private static final long OPTIMAL_MAX_NUMBER_PIXELS = 4096 * 4096 * 3;
	
	private TileFactory(ModelDescriptor descriptor) {
		this.descriptor = descriptor;
	}

	public static TileFactory init(ModelDescriptor descriptor) {
		return new TileFactory(descriptor);
	}
	
	public long[] getOptimalTileSize(String tensorName, long[] dims, String inputAxesOrder) {
		// TODO TensorSpec tensor = descriptor.findInputTensor(tensorName);
		TensorSpecV05 tensor = null;
		boolean tiling = this.descriptor.isTilingAllowed();
		int[] halo = descriptor.getTotalHalo();
		int[] min = tensor.getMinTileSizeArr();
		int[] step = tensor.getTileStepArr();
		double[] scale = tensor.getTileScaleArr();
		
    	long[] patch = new long[inputAxesOrder.length()];
    	String seqSizeAxesUpper = inputAxesOrder.toUpperCase();
    	seqSizeAxesUpper = seqSizeAxesUpper.replace("T", "B");
    	String[] axesArr = tensor.getAxesOrder().toUpperCase().split("");
    	
    	
    	
		for (int ii = 0; ii < axesArr.length; ii ++) {
			int ind = seqSizeAxesUpper.indexOf(axesArr[ii]);
			int size = (int) dims[ind];
			if (step[ii] != 0 && tiling) {
				patch[ii] = (int)Math.ceil((size + 2 * halo[ii]) / step[ii]) * step[ii];
				// The patch cannot be 3 times bigger than the size of the image, because then
				// mirroring would stop working
				if (patch[ii] > 3 * size && (patch[ii] - step[ii]) >= min[ii]) 
					patch[ii] = patch[ii] - step[ii];
				if (patch[ii] > 3 * size && ((int)Math.ceil((double)size / (double)step[ii]) * step[ii]) >= min[ii]) 
					patch[ii] = (int)Math.ceil((double)size / (double)step[ii]) * step[ii];
			} else if (step[ii] != 0 && !tiling){
				// If tiling is not allowed and the step can vary, the patch size for that
				// dimension is left to -1 and then calculated automatically after pre-processing
				// just in case pre-processing changes the image size
				patch[ii] = -1;
			} else if (step[ii] == 0){
				patch[ii] = min[ii];
			}
		}
		
		if (!tiling || Arrays.stream(tensor.getTileStepArr()).allMatch(i -> i == 0))
			return patch;
		long totPix = 1;
		for (long ii : patch) totPix *= (long) ii;
		
		List<String> affectedTensors = tensor.getAxesInfo().getAxesList().stream()
										.map(i -> i.getReferenceTensor()).collect(Collectors.toList());
		
		long outputTotByteSize = calculateByteSizeOfAffectedOutput(tensor.getAxesOrder(), patch, affectedTensors);
		
		if (totPix < OPTIMAL_MAX_NUMBER_PIXELS && outputTotByteSize < Integer.MAX_VALUE)
			return patch;
		
		long minPix = 1;
		for (int ii : tensor.getMinTileSizeArr()) minPix *= (long) ii;
		if (minPix > OPTIMAL_MAX_NUMBER_PIXELS)
			return Arrays.stream(min).mapToLong(i -> i).toArray();

		double ratioSize = (double) OPTIMAL_MAX_NUMBER_PIXELS / (double) totPix;
		double ratioByte = (double) Integer.MAX_VALUE / (double) outputTotByteSize;
		
		double ratio = Math.min(ratioSize, ratioByte);
		
		for (int ii = 0; ii < axesArr.length; ii ++) {
			if (step[ii] == 0) continue;
			long prevTile = patch[ii];
			long nTot = totPix / prevTile;
			if ((prevTile * ratio < min[ii]) && (min[ii] < 100) && (min[ii] != 1) && (min[ii] != 0))
				patch[ii] = (int)Math.ceil((double)100 / (double)step[ii]) * step[ii];
			else if (prevTile * ratio < min[ii])
				patch[ii] = min[ii];
			else 
				patch[ii] = (long) (Math.floor((prevTile * ratio - min[ii]) / step[ii]) * step[ii] + min[ii]);
			totPix = nTot * patch[ii];
			ratioSize = (double) OPTIMAL_MAX_NUMBER_PIXELS / (double) totPix;
			ratioByte = (double) Integer.MAX_VALUE / (double) calculateByteSizeOfAffectedOutput(tensor.getAxesOrder(), patch, affectedTensors);
			ratio = Math.min(ratioSize, ratioByte);
			if (ratio > 1)
				break;
		}
		return patch;
	}
    
    private long calculateByteSizeOfAffectedOutput(String inputAxes, long[] inputSize, List<String> affectedOutputs) {
    	if (affectedOutputs == null || affectedOutputs.size() == 0) return 0;
        inputSize = PatchGridCalculator.arrayToWantedAxesOrderAddOnes(inputSize, inputAxes, affectedOutput.axes);
        int[] outputSize = new int[inputSize.length];
        if (!affectedOutput.shape.isFixedSize()) {
        	float[] scale = affectedOutput.shape.getScale();
        	float[] offset = affectedOutput.shape.getOffset();
        	for (int i = 0; i < scale.length; i ++) {
        		outputSize[i] = (int) (inputSize[i] * scale[i] + offset[i] * 2);
        	}
        } else {
        	outputSize = affectedOutput.shape.getTileRecomendedSize();
        }
        
        long flatSize = 1;
        for (int i : outputSize) flatSize *= (long) i;
        if (affectedOutput.dataType.toLowerCase().equals("float32")
        		|| affectedOutput.dataType.toLowerCase().equals("int32")
        		|| affectedOutput.dataType.toLowerCase().equals("uint32"))
        	flatSize *= 4;
        else if (affectedOutput.dataType.toLowerCase().equals("int16")
        		|| affectedOutput.dataType.toLowerCase().equals("uint16"))
        	flatSize *= 2;
        else if (affectedOutput.dataType.toLowerCase().equals("int64")
        		|| affectedOutput.dataType.toLowerCase().equals("float64"))
        	flatSize *= 8;
        return flatSize;
    }
	
	public void validateTileSize(String tensorName, long[] dims, String inputAxesOrder) {
		
	}
	
	public void getTileSizeForNTiles(int nTiles, String tensorName, long[] dims, String inputAxesOrder) {
		
	}
}
