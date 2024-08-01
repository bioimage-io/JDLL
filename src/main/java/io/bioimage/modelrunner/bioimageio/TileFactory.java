package io.bioimage.modelrunner.bioimageio;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
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
	
	private long[] getOptimalTileSize(TensorSpec tensor, String inputAxesOrder, long[] dims) {
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
		return patch;
	}
	
	public List<ImageInfo> getOptimalTileSize(List<ImageInfo> inputInfo) {
		boolean tiling = this.descriptor.isTilingAllowed();
		List<ImageInfo> firstIterationInputs = new ArrayList<ImageInfo>();
		for (TensorSpec tt : this.descriptor.getInputTensors()) {
			ImageInfo im = inputInfo.stream()
					.filter(ii -> ii.getTensorName().equals(tt.getTensorID())).findFirst().orElse(null);
			if (im == null) 
				throw new IllegalArgumentException("No data was provided for input tensor: " + tt.getTensorID());
			
			long[] tileSize = getOptimalTileSize(tt, im.getAxesOrder(), im.getDimensions());
			
			firstIterationInputs.add(new ImageInfo(im.getTensorName(), im.getAxesOrder(), tileSize));
		}
		
		if (!tiling)
			return firstIterationInputs;
		
		List<TensorSpec> affectedTensors = this.descriptor.getOutputTensors().stream()
				.filter(ot -> {
					return ot.getAxesInfo().getAxesList().stream()
					.filter(ax -> ax.getReferenceTensor() != null)
					.findFirst().orElse(null) != null;
		}).collect(Collectors.toList());

		List<ImageInfo> secondIterationInputs = new ArrayList<ImageInfo>();
		for (int i = 0; i < firstIterationInputs.size(); i ++) {
			TensorSpec tensor = descriptor.findInputTensor(firstIterationInputs.get(i).getTensorName());
			if (Arrays.stream(tensor.getTileStepArr()).allMatch(ii -> ii == 0)) {
				secondIterationInputs.add(firstIterationInputs.get(i));
			}
		}
		if (firstIterationInputs.size() == firstIterationInputs.size())
			return secondIterationInputs;
		
		
		List<Long> outputTotByteSizes = calculateByteSizeOfAffectedOutput(affectedTensors, null, null);
		return checkOutputSize(firstIterationInputs, affectedTensors, outputTotByteSizes);
	}
	
	
	private List<ImageInfo> checkOutputSize(List<ImageInfo> inputs, List<TensorSpec> affected, List<Long> outByteSizes) {
		List<Long> totInPixels = inputs.stream().map(in -> Arrays.stream(in.getDimensions()).reduce(1, (x, y) -> x * y)).collect(Collectors.toList());
			
		
		if (totInPixels.stream().filter(oo -> oo > OPTIMAL_MAX_NUMBER_PIXELS).findFirst().orElse(null) == null 
				&& outByteSizes.stream().filter(oo -> oo > Integer.MAX_VALUE).findFirst().orElse(null) == null)
			return inputs;


		List<Double> inRatio = totInPixels.stream().map(ss -> (double) OPTIMAL_MAX_NUMBER_PIXELS / (double) ss).collect(Collectors.toList());
		List<Double> outRatio = outByteSizes.stream().map(ss -> (double) Integer.MAX_VALUE / (double) ss).collect(Collectors.toList());
		
		while(Collections.min(inRatio) < 1) {
			Integer argmin = null;
			List<Integer> sortedIndices = IntStream.range(0, inRatio.size())
				    .boxed()
				    .sorted(Comparator.comparing(inRatio::get))
				    .collect(Collectors.toList());
			for (Integer ind : sortedIndices) {
				TensorSpec tt = this.descriptor.findInputTensor(inputs.get(ind).getTensorName());
				if (Arrays.stream(tt.getTileStepArr()).allMatch(ii -> ii == 0))
					continue;
				argmin = ind;
				break;
			}
			if (argmin == null) break;
			
			Double startingRatio = inRatio.get(argmin);
			ImageInfo in = inputs.get(argmin);
			long[] dims = in.getDimensions();
			TensorSpec tt = this.descriptor.findInputTensor(in.getTensorName());
			int c = 0;
			for (String ax : in.getAxesOrder().split("")) {
				Axis axis = tt.getAxesInfo().getAxis(ax);
				if (axis.getStep() == 0) continue;
				long nTot = totInPixels.get(argmin) / in.getDimensions()[c];
				if ((in.getDimensions()[c] * inRatio.get(argmin) < axis.getMin()) && (axis.getMin() > 1)) {
					in.getDimensions()[c] = (int)Math.ceil((double) 100 / (double) axis.getStep()) * axis.getStep();
				} else if (in.getDimensions()[c] * inRatio.get(argmin) < axis.getMin()) {
					in.getDimensions()[c] = axis.getMin();
				} else {
					in.getDimensions()[c] = (long) (Math.floor((in.getDimensions()[c] * inRatio.get(argmin) - axis.getMin()) / axis.getStep()) * axis.getStep() + axis.getMin());
				}
				totInPixels.set(argmin, nTot * in.getDimensions()[c]);
				inRatio = totInPixels.stream().map(ss -> (double) OPTIMAL_MAX_NUMBER_PIXELS / (double) ss).collect(Collectors.toList());
				
				if (startingRatio == inRatio.get(argmin))
					break;
			}
		}
		
		outByteSizes = calculateByteSizeOfAffectedOutput(affected, null, null);
		outRatio = outByteSizes.stream().map(ss -> (double) Integer.MAX_VALUE / (double) ss).collect(Collectors.toList());
		
		if (Collections.min(outRatio) < 1 && Collections.min(inRatio) < 1 )
			throw new IllegalArgumentException("The input and/or ouput dimensions of the tensors specified by the current model are to big. JDLL is not able to run them.");
		
		while(Collections.min(outRatio) < 1) {
			final List<Double> finalOutRatio = new ArrayList<>(outRatio);
			int argmin = IntStream.range(0, finalOutRatio.size())
			    .reduce((i, j) -> finalOutRatio.get(i) < finalOutRatio.get(j) ? i : j)
			    .getAsInt();
			Double oldRatio = outRatio.get(argmin);
			
			TensorSpec tt = this.descriptor.getOutputTensors().get(argmin);
			for (Axis ax : tt.getAxesInfo().getAxesList()) {
				if (ax.getReferenceTensor() == null)
					continue;
				TensorSpec inputT = this.descriptor.findInputTensor(ax.getReferenceTensor());
				ImageInfo im = inputs.stream().filter(in -> in.getTensorName().equals(inputT.getTensorID())).findFirst().orElse(null);
				String refAxis = ax.getReferenceAxis();
				int index = im.getAxesOrder().indexOf(refAxis);
				Axis inAx = inputT.getAxesInfo().getAxis(refAxis);
				long size = im.getDimensions()[index];
				
				if ((size * outRatio.get(argmin) < inAx.getMin()) && (inAx.getMin() > 1)) {
					im.getDimensions()[index] = (int)Math.ceil((double) 100 / (double) inAx.getStep()) * inAx.getStep();
				} else if (size * outRatio.get(argmin) < inAx.getMin()) {
					im.getDimensions()[index] = inAx.getMin();
				} else {
					im.getDimensions()[index] = (long) (Math.floor((size * outRatio.get(argmin) - inAx.getMin()) / inAx.getStep()) * inAx.getStep() + inAx.getMin());
				}
				double change = (size * ax.getScale() + 2 * ax.getOffset()) / (im.getDimensions()[index] * ax.getScale() + 2 * ax.getOffset());
				outRatio.set(argmin, outRatio.get(argmin) * change);
				if (outRatio.get(argmin) > 1)
					break;
			}
			if (outRatio.get(argmin) == oldRatio)
				break;
		}
		
		if (Collections.min(outRatio) < 1)
			throw new IllegalArgumentException("Due to the model specifications, the size of one of the output tensors exceeds the limit of tensor size in JDLL: " + Integer.MAX_VALUE);
		return null;
	}
	
	public void validateTileSize(String tensorName, long[] dims, String inputAxesOrder) {
		
	}
	
	public void getTileSizeForNTiles(int nTiles, String tensorName, long[] dims, String inputAxesOrder) {
		
	}
    
    private List<Long> calculateByteSizeOfAffectedOutput(List<TensorSpec> inputTensors, List<long[]> inputSize, List<String> affectedOutputs) {
    	if (affectedOutputs == null || affectedOutputs.size() == 0) 
    		return LongStream.range(0, affectedOutputs.size()).map(i -> 0L).boxed().collect(Collectors.toList());
    	List<String> names = inputTensors.stream().map(t -> t.getTensorID()).collect(Collectors.toList());
    	List<String> axesOrders = inputTensors.stream().map(t -> t.getAxesOrder()).collect(Collectors.toList());
    	List<TensorSpec> outputTensors = this.descriptor.getOutputTensors();
    	outputTensors = outputTensors.stream()
    			.filter(t -> {
    				return t.getAxesInfo().getAxesList().stream()
    						.filter(tt -> names.contains(tt.getReferenceTensor())).findFirst().orElse(null) != null;
    			}).collect(Collectors.toList());
    	
    	List<long[]> outTiles = outputTensors.stream()
    			.map(t -> new long[t.getAxesInfo().getAxesList().size()]).collect(Collectors.toList());
    	
    	for (int i = 0; i < outputTensors.size(); i ++) {
    		TensorSpec tt = outputTensors.get(i);
    		for (int j = 0; j < outputTensors.get(i).getAxesInfo().getAxesList().size(); j ++) {
    			Axis ax = tt.getAxesInfo().getAxesList().get(j);
    			if (ax.getStep() == 0) {
    				outTiles.get(i)[j] = ax.getMin();
    				continue;
    			}
    			String refName = ax.getReferenceTensor();
    			String refAxisStr = ax.getReferenceAxis();
    			TensorSpec refTensor = inputTensors.get(names.indexOf(refName));
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
