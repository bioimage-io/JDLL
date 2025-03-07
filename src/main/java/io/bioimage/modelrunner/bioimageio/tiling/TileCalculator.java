/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.bioimageio.tiling;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import io.bioimage.modelrunner.bioimageio.description.Axis;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.bioimageio.tiling.TileInfo;
import io.bioimage.modelrunner.utils.Constants;

public class TileCalculator {
	
	private final ModelDescriptor descriptor;
    
    private static final long OPTIMAL_MAX_NUMBER_PIXELS = 4096 * 4096 * 3;
	
	private TileCalculator(ModelDescriptor descriptor) {
		this.descriptor = descriptor;
	}

	public static TileCalculator init(ModelDescriptor descriptor) {
		return new TileCalculator(descriptor);
	}
	
	// TODO what to do when the axes order do not coincide
	private long[] getOptimalTileSize(TensorSpec tensor, String inputAxesOrder, long[] dims) {
		boolean tiling = this.descriptor.isTilingAllowed();
		int[] halo = tensor.getAxesInfo().getHaloArr();
		int[] min = tensor.getMinTileSizeArr();
		int[] step = tensor.getTileStepArr();
		
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
			} else if (step[ii] == 0 && min[ii] == -1){
				patch[ii] = size;
			} else if (step[ii] == 0){
				patch[ii] = min[ii];
			}
		}
		return patch;
	}
	
	public List<TileInfo> getOptimalTileSize(List<ImageInfo> inputInfo) {
		boolean tiling = this.descriptor.isTilingAllowed();
		List<TileInfo> firstIterationInputs = new ArrayList<TileInfo>();
		for (TensorSpec tt : this.descriptor.getInputTensors()) {
			ImageInfo im = inputInfo.stream()
					.filter(ii -> ii.getTensorName().equals(tt.getName())).findFirst().orElse(null);
			if (im == null) 
				throw new IllegalArgumentException("No data was provided for input tensor: " + tt.getName());
			
			long[] tileSize = getOptimalTileSize(tt, im.getAxesOrder(), im.getDimensions());
			
			firstIterationInputs.add(TileInfo.build(tt.getName(), im.getDimensions(), im.getAxesOrder(), tileSize, im.getAxesOrder()));
		}
		
		if (!tiling)
			return firstIterationInputs;
		
		List<TensorSpec> affectedTensors = this.descriptor.getOutputTensors().stream()
				.filter(ot -> {
					return ot.getAxesInfo().getAxesList().stream()
					.filter(ax -> ax.getReferenceTensor() != null)
					.findFirst().orElse(null) != null;
		}).collect(Collectors.toList());

		List<TileInfo> secondIterationInputs = new ArrayList<TileInfo>();
		for (int i = 0; i < firstIterationInputs.size(); i ++) {
			TensorSpec tensor = descriptor.findInputTensor(firstIterationInputs.get(i).getName());
			if (Arrays.stream(tensor.getTileStepArr()).allMatch(ii -> ii == 0)) {
				secondIterationInputs.add(firstIterationInputs.get(i));
			}
		}
		if (firstIterationInputs.size() == secondIterationInputs.size())
			return secondIterationInputs;
		
		
		List<Long> outputTotByteSizes = calculateByteSizeOfAffectedOutput(affectedTensors, firstIterationInputs);
		return checkOutputSize(firstIterationInputs, affectedTensors, outputTotByteSizes);
	}
	
	
	private List<TileInfo> checkOutputSize(List<TileInfo> inputs, List<TensorSpec> affected, List<Long> outByteSizes) {
		List<Long> totInPixels = inputs.stream().map(in -> Arrays.stream(in.getTileDims()).reduce(1, (x, y) -> x * y)).collect(Collectors.toList());
			
		
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
				TensorSpec tt = this.descriptor.findInputTensor(inputs.get(ind).getName());
				if (Arrays.stream(tt.getTileStepArr()).allMatch(ii -> ii == 0))
					continue;
				argmin = ind;
				break;
			}
			if (argmin == null) break;
			
			Double startingRatio = inRatio.get(argmin);
			TileInfo in = inputs.get(argmin);
			TensorSpec tt = this.descriptor.findInputTensor(in.getName());
			int c = 0;
			for (String ax : in.getTileAxesOrder().split("")) {
				Axis axis = tt.getAxesInfo().getAxis(ax);
				if (axis.getStep() == 0) continue;
				long nTot = totInPixels.get(argmin) / in.getTileDims()[c];
				if ((in.getTileDims()[c] * inRatio.get(argmin) < axis.getMin()) && (axis.getMin() > 1)) {
					in.getTileDims()[c] = (int)Math.ceil((double) 100 / (double) axis.getStep()) * axis.getStep();
				} else if (in.getTileDims()[c] * inRatio.get(argmin) < axis.getMin()) {
					in.getTileDims()[c] = axis.getMin();
				} else {
					in.getTileDims()[c] = (long) (Math.floor((in.getTileDims()[c] * inRatio.get(argmin) - axis.getMin()) / axis.getStep()) * axis.getStep() + axis.getMin());
				}
				totInPixels.set(argmin, nTot * in.getTileDims()[c]);
				inRatio = totInPixels.stream().map(ss -> (double) OPTIMAL_MAX_NUMBER_PIXELS / (double) ss).collect(Collectors.toList());
				
				if (startingRatio == inRatio.get(argmin))
					break;
			}
		}
		
		outByteSizes = calculateByteSizeOfAffectedOutput(affected, null);
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
				TileInfo im = inputs.stream().filter(in -> in.getName().equals(inputT.getName())).findFirst().orElse(null);
				String refAxis = ax.getReferenceAxis();
				int index = im.getTileAxesOrder().indexOf(refAxis);
				Axis inAx = inputT.getAxesInfo().getAxis(refAxis);
				long size = im.getTileDims()[index];
				
				if ((size * outRatio.get(argmin) < inAx.getMin()) && (inAx.getMin() > 1)) {
					im.getTileDims()[index] = (int)Math.ceil((double) 100 / (double) inAx.getStep()) * inAx.getStep();
				} else if (size * outRatio.get(argmin) < inAx.getMin()) {
					im.getTileDims()[index] = inAx.getMin();
				} else {
					im.getTileDims()[index] = (long) (Math.floor((size * outRatio.get(argmin) - inAx.getMin()) / inAx.getStep()) * inAx.getStep() + inAx.getMin());
				}
				double change = (size * ax.getScale() + 2 * ax.getOffset()) / (im.getTileDims()[index] * ax.getScale() + 2 * ax.getOffset());
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
	
	public void getTilesForNPixels(String tensorName, long[] dims, String inputAxesOrder) {
		
	}
	
	public void getForNTiles(int nTiles, String tensorName, long[] dims, String inputAxesOrder) {
		
	}
    
    private List<Long> calculateByteSizeOfAffectedOutput(List<TensorSpec> outputTensors, List<TileInfo> inputSize) {
    	if (outputTensors == null || outputTensors.size() == 0) 
    		return new ArrayList<Long>();
    	
    	List<long[]> outTiles = outputTensors.stream()
    			.map(t -> new long[t.getAxesInfo().getAxesList().size()]).collect(Collectors.toList());
    	
    	for (int i = 0; i < outputTensors.size(); i ++) {
    		TensorSpec tt = outputTensors.get(i);
    		ArrayList<String> referencesList = new ArrayList<String>();
    		for (int j = 0; j < outputTensors.get(i).getAxesInfo().getAxesList().size(); j ++) {
    			Axis ax = tt.getAxesInfo().getAxesList().get(j);
    			String refName = ax.getReferenceTensor();
    			if (refName == null && ax.getMin()!= 0) {
    				outTiles.get(i)[j] = ax.getMin();
    				continue;
    			} else if (refName == null) {
    				outTiles.get(i)[j] = -1;
    				continue;
    			}
				referencesList.add(refName);
    			String refAxisStr = ax.getReferenceAxis();
    			TensorSpec refTensor = descriptor.findInputTensor(refName);
    			long[] refTileSize = inputSize.stream()
    					.filter(tile -> tile.getName().equals(refName)).findFirst().orElse(null).getTileDims();
    			String axesOrder = refTensor.getAxesOrder();
    			outTiles.get(i)[j] = 
    					(long) (refTileSize[axesOrder.indexOf(refAxisStr)] * ax.getScale() + 2 * ax.getOffset());
    		}
    		if (referencesList.stream().distinct().count() != 1)
    			throw new IllegalArgumentException(""
						+ "Model specs too complex for JDLL. "
						+ "Please contact the team and create and issue attaching the rdf.yaml file"
						+ " so we can troubleshoot at: " + Constants.ISSUES_LINK);
    		else {
    			for (int j = 0; j < outputTensors.get(i).getAxesInfo().getAxesList().size(); j ++) {
    				if (outTiles.get(i)[j] != -1)
    					continue;
    				TensorSpec refInput = this.descriptor.findInputTensor(referencesList.get(0));
    				int ind = refInput.getAxesOrder().indexOf(outputTensors.get(i).getAxesInfo().getAxesList().get(j).getAxis());
    				if (ind == -1)
    	    			throw new IllegalArgumentException(""
    							+ "Model specs too complex for JDLL. "
    							+ "Please contact the team and create and issue attaching the rdf.yaml file"
    							+ " so we can troubleshoot at: " + Constants.ISSUES_LINK);
        			long[] refTileSize = inputSize.stream()
        					.filter(tile -> tile.getName().equals(referencesList.get(0))).findFirst().orElse(null).getTileDims();
    				outTiles.get(i)[j] = refTileSize[ind];
    			}
    		}
    	}
        
    	List<Long> flatSizes = outTiles.stream().map(arr -> {
								                    long a = 1L;
								                    for (long l : arr) a *= l;
								                    return a;
								                }).collect(Collectors.toList());

    	for (int i = 0; i < flatSizes.size(); i ++) {
            if (outputTensors.get(i).getDataType().toLowerCase().equals("float32")
            		|| outputTensors.get(i).getDataType().toLowerCase().equals("int32")
            		|| outputTensors.get(i).getDataType().toLowerCase().equals("uint32"))
            	flatSizes.set(i, flatSizes.get(i) * 4);
            else if (outputTensors.get(i).getDataType().toLowerCase().equals("int16")
            		|| outputTensors.get(i).getDataType().toLowerCase().equals("uint16"))
            	flatSizes.set(i, flatSizes.get(i) * 2);
            else if (outputTensors.get(i).getDataType().toLowerCase().equals("int64")
            		|| outputTensors.get(i).getDataType().toLowerCase().equals("float64"))
            	flatSizes.set(i, flatSizes.get(i) * 8);
    	}
        return flatSizes;
    }
}
