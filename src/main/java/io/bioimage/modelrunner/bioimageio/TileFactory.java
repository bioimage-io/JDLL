package io.bioimage.modelrunner.bioimageio;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.axes.axis.Axis;
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
		TensorSpec tensor = null;
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
		
		List<Long> outputTotByteSizes = calculateByteSizeOfAffectedOutput(null, null, affectedTensors);
		
		if (totPix < OPTIMAL_MAX_NUMBER_PIXELS 
				&& outputTotByteSizes.stream().filter(oo -> oo > Integer.MAX_VALUE).findFirst().orElse(null) == null)
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
	
	public void validateTileSize(String tensorName, long[] dims, String inputAxesOrder) {
		
	}
	
	public void getTileSizeForNTiles(int nTiles, String tensorName, long[] dims, String inputAxesOrder) {
		
	}
    
    private List<Long> calculateByteSizeOfAffectedOutput(List<TensorSpecV05> inputTensors, List<long[]> inputSize, List<String> affectedOutputs) {
    	if (affectedOutputs == null || affectedOutputs.size() == 0) 
    		return LongStream.range(0, affectedOutputs.size()).map(i -> 0L).boxed().collect(Collectors.toList());
    	List<String> names = inputTensors.stream().map(t -> t.getTensorID()).collect(Collectors.toList());
    	List<String> axesOrders = inputTensors.stream().map(t -> t.getAxesOrder()).collect(Collectors.toList());
    	List<TensorSpecV05> outputTensors = null; // TODO
    	outputTensors = outputTensors.stream()
    			.filter(t -> {
    				return t.getAxesInfo().getAxesList().stream()
    						.filter(tt -> names.contains(tt.getReferenceTensor())).findFirst().orElse(null) != null;
    			}).collect(Collectors.toList());
    	
    	List<long[]> outTiles = outputTensors.stream()
    			.map(t -> new long[t.getAxesInfo().getAxesList().size()]).collect(Collectors.toList());
    	
    	for (int i = 0; i < outputTensors.size(); i ++) {
    		TensorSpecV05 tt = outputTensors.get(i);
    		for (int j = 0; j < outputTensors.get(i).getAxesInfo().getAxesList().size(); j ++) {
    			Axis ax = tt.getAxesInfo().getAxesList().get(j);
    			if (ax.getStep() == 0) {
    				outTiles.get(i)[j] = ax.getMin();
    				continue;
    			}
    			String refName = ax.getReferenceTensor();
    			String refAxisStr = ax.getReferenceAxis();
    			TensorSpecV05 refTensor = inputTensors.get(names.indexOf(refName));
    			long[] refTileSize = inputSize.get(names.indexOf(refName));
    			String axesOrder = axesOrders.get(names.indexOf(refName));
    			Axis refAxis = refTensor.getAxesInfo().getAxis(refAxisStr);
    			outTiles.get(i)[j] = 
    					(long) (refTileSize[axesOrder.indexOf(refAxisStr)] * refAxis.getScale() + refAxis.getOffset());
    		}
    	}
        
    	List<Long> flatSizes = LongStream.range(0, outTiles.size()).map(i -> 1L).boxed().collect(Collectors.toList());

    	for (int i = 0; i < flatSizes.size(); i ++) {
            if (outputTensors.get(i).getDataType().toLowerCase().equals("float32")
            		|| outputTensors.get(i).getDataType().toLowerCase().equals("int32")
            		|| outputTensors.get(i).getDataType().toLowerCase().equals("uint32"))
            	flatSizes.set(i, flatSizes.get(i) * 8);
            else if (outputTensors.get(i).getDataType().toLowerCase().equals("int16")
            		|| outputTensors.get(i).getDataType().toLowerCase().equals("uint16"))
            	flatSizes.set(i, flatSizes.get(i) * 2);
            else if (outputTensors.get(i).getDataType().toLowerCase().equals("int64")
            		|| outputTensors.get(i).getDataType().toLowerCase().equals("float64"))
            	flatSizes.set(i, flatSizes.get(i) * 4);
            for (long j : outTiles.get(i)) 
            	flatSizes.set(i, flatSizes.get(i) * j);
    	}
        return flatSizes;
    }
}
