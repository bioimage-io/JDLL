/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.tiling;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.tensor.Tensor;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.Type;

/**
 * A calculator for the size of each patch and the patch grid associated to input images when applying a given TensorFlow model.
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando 
 */
public class PatchGridCalculator implements Callable<List<PatchSpec>>
{

    private ModelDescriptor descriptor;
    private Map<String, Object> inputValuesMap;

    /**
     * Class to calculate the patch specifications given a series of inputs
     * and their specifications
     * @param descriptor
     * 	the specifications of each input
     * @param inputValuesMap
     * 	mapt containing the input images associated to their input tensors
     */
    private PatchGridCalculator(ModelDescriptor descriptor, Map<String, Object> inputValuesMap)
    {
    	this.descriptor = descriptor;
    	this.inputValuesMap = inputValuesMap;
    }
    
    /**
     * Create the patch specifications for the provided model
     * @param model
     * 	model that needs patching specs
     * @param inputValuesMap
     * 	mapt containing the input images associated to their input tensors
     * @return the object that creates a list of patch specs for each tensor
     */
    public static PatchGridCalculator build(DeepLearningModel model, Map<String, Object> inputValuesMap) {
    	 return new PatchGridCalculator(model.getDescriptor(), inputValuesMap);
     }
    
    /**
     * Create the patch specifications for the model spces
     * @param model
     * 	model specs
     * @param inputValuesMap
     * 	mapt containing the input images associated to their input tensors
     * @return the object that creates a list of patch specs for each tensor
     */
    public static PatchGridCalculator build(ModelDescriptor model, Map<String, Object> inputValuesMap) {
    	 return new PatchGridCalculator(model, inputValuesMap);
     }

    /**
     * Computes the patch size adapted for the input sequence using the model tensor specification.
     * 
     * @return The patch specifications to use for this model and the input sequence.
     * @throws IllegalArgumentException if one tensor that allows tiling needs more patches
     * 	in any given axis than the others
     */
    @Override
    public List<PatchSpec> call() throws RuntimeException, IllegalArgumentException
    {
    	List<TensorSpec> inputTensors = findInputImageTensorSpec();
        List<Object> inputImages = findModelInputImages(inputTensors);
        List<PatchSpec> listPatchSpecs = computePatchSpecsForEveryTensor(inputTensors, inputImages);
        // Check that the obtained patch specs are not going to cause errors
        checkPatchSpecs(listPatchSpecs);
        return listPatchSpecs;
    }
    
    /**
     * Check that the relationship between tensor and image is the same for all the tensors
     * that allow tiling
     * @param listPatchSpecs
     * 	specs for each of the tensors
     * @throws IllegalArgumentException if one tensor that allows tiling needs more patches
     * 	in any given axis than the others
     */
    public void checkPatchSpecs(List<PatchSpec> listPatchSpecs) throws IllegalArgumentException {
    	int[] grid = null;
    	String firstName = null;
    	for (PatchSpec spec : listPatchSpecs) {
    		int[] nGrid = spec.getPatchGridSize();
    		TensorSpec tt = this.descriptor.findInputTensor(spec.getTensorName());
    		if (grid == null && tt.getTiling()) {
    			grid = nGrid;
    			firstName = spec.getTensorName();
    		}
    		if (tt.getTiling() && !compareTwoArrays(nGrid, grid)){
    			throw new IllegalArgumentException("All the input images must be processed with the same number of patches.\n"
						+ "The relationship between the patch size and image size should be the same for every input that allows patching/tiling.\n"
						+ "Tensors '" + firstName + "' and '" + spec.getTensorName() + "' need different number of patches to "
						+ "process their images and that is not supported at the moment.");
    		}
    	}
    	
    }
    
    /**
     * Check whether or not two arrays are equal
     * @param arr1
     * 	one array
     * @param arr2
     * 	another array
     * @return true if the arrays are the same or false otherwise
     */
    public static boolean compareTwoArrays(int[] arr1, int[] arr2) {
    	return (arr1.length == arr2.length &&
    	        IntStream.range(0, arr1.length)
    	                 .allMatch(i -> arr1[i] == arr2[i]));
    }

    /**
     * Get the input tensors that correspond to images
     * @return list of tensor specs corresponding to each of the input image tensors
     */
    private List<TensorSpec> findInputImageTensorSpec()
    {
        return this.descriptor.getInputTensors().stream().filter(tr -> tr.getType() == "image")
                .collect(Collectors.toList());
    }

    /**
     * Get the list of sequences that correspond to each of the tensors
     * @param inputTensorSpec
     * 	the list of input tensors
     * @return the list of input images
     * @throws NoSuchElementException if there is an image missing for each of the input tensors
     */
    private List<Object> findModelInputImages(List<TensorSpec> inputTensorSpec) throws NoSuchElementException
    {
    	List<Object> seqList = inputTensorSpec.stream()
							    	.filter(t -> inputValuesMap.get(t.getName()) != null)
							    	.map(im -> inputValuesMap.get(im.getName()))
									.collect(Collectors.toList());
        if (seqList.size() != inputTensorSpec.size()) {
        	List<String> missing = inputTensorSpec.stream()
							    	.filter(t -> inputValuesMap.get(t.getName()) == null)
							    	.map(im -> im.getName())
									.collect(Collectors.toList());
        	String errMsg = "Could not find any input Icy Sequence, Icy Tensor or NDArray for the following tensors:\n";
        	for (int i = 0; i < missing.size(); i ++) {
        		errMsg += " -" + missing.get(i);
        	}
            throw new NoSuchElementException(errMsg);
        }

        return seqList;
    }
    
    /**
     * Create list of patch specifications for every tensor aking into account the
     * corresponding image
     * @param tensors
     * 	the tensor information
     * @param images
     * 	the images corresponding to each tensor
     * @return the list of patch specifications for each tensor
     */
    private List<PatchSpec> computePatchSpecsForEveryTensor(List<TensorSpec> tensors, List<Object> images){
    	List<PatchSpec> patchInfoList = new ArrayList<PatchSpec>();
    	for (int i = 0; i < tensors.size(); i ++)
    		patchInfoList.add(computePatchSpecs(tensors.get(i), images.get(i)));
    	return patchInfoList;
    }

    /**
     * Compute the patch details needed to perform the tiling strategy. The calculations
     * obtain the input patch, the padding needed at each side and teh number of patches
     * needed for every tensor
     * @param inputTensorSpec
     * 	specs of the tensor
     * @param inputSequence
     * 	object that is going to be patched
     * @return an object containing the specs needed to perform patching for the particular tensor
     * @throws IllegalArgumentException if the JAva type of the input object is not among the 
     * 	allowed ones ({@link Sequence}, {@link NDArray} or {@link Tensor})
     */
    private <T extends Type<T>> PatchSpec computePatchSpecs(TensorSpec inputTensorSpec, Object inputObject)
    		throws IllegalArgumentException {
    	if (inputObject instanceof RandomAccessibleInterval) {
    		return computePatchSpecs(inputTensorSpec, (RandomAccessibleInterval<T>) inputObject);
    	} else if(inputObject instanceof Tensor ) {
    		return computePatchSpecs(inputTensorSpec, ((Tensor) inputObject).getData());
    	} else {
    		throw new IllegalArgumentException("Input tensor '" + inputTensorSpec.getName()
    		+ "' is not represented with a Java type (" + inputObject.getClass().toString()
    		+ ") that JDLL can pass to a model. JDLL can only handle inputs to the model as:" + System.lineSeparator()
    				+ "- " + RandomAccessibleInterval.class.toString() + System.lineSeparator()
    				+ "- " + Tensor.class.toString() + System.lineSeparator());
    	}
    }

    /**
     * Compute the patch details needed to perform the tiling strategy. The calculations
     * obtain the input patch, the padding needed at each side and the number of patches
     * needed for every tensor.
     * To make it standard for all axes order and patches, the calculations are always done
     * using the "xyczb" axes order
     * @param inputTensorSpec
     * 	specs of the tensor
     * @param inputSequence
     * 	sequence that is going to be patched
     * @param inputPatch
     * 	input patch to the model
     * @return an object containing the specs needed to perform patching for the particular tensor
     */
    private <T extends Type<T>> PatchSpec computePatchSpecs(TensorSpec inputTensorSpec, RandomAccessibleInterval<T> inputSequence)
    {
    	String processingAxesOrder = "xyczb";
        int[] inputPatchSize = arrayToWantedAxesOrderAddOnes(inputTensorSpec.getProcessingPatch(),
				        										inputTensorSpec.getAxesOrder(), 
				        										processingAxesOrder);
        int[][] paddingSize = new int[2][5];
        // REgard that the input halo represents the output halo + offset 
        // and must be divisible by 0.5. 
        float[] halo = arrayToWantedAxesOrderAddZeros(inputTensorSpec.getHalo(),
												        		inputTensorSpec.getAxesOrder(), 
																processingAxesOrder);
        if (!descriptor.isPyramidal() && inputTensorSpec.getTiling()) {
        	// In the case that padding is asymmetrical, the left upper padding has the extra pixel
            for (int i = 0; i < halo.length; i ++) {paddingSize[0][i] = (int) Math.ceil(halo[i]);}
            // In the case that padding is asymmetrical, the right bottom padding has one pixel less
            for (int i = 0; i < halo.length; i ++) {paddingSize[1][i] = (int) Math.floor(halo[i]);}
            
        }
        long[] shapeLong = inputSequence.dimensionsAsLongArray();
        int[] shapeInt = new int[shapeLong.length];
        for (int i = 0; i < shapeInt.length; i ++) {shapeInt[i] = (int) shapeLong[i];}
        int[] inputSequenceSize = arrayToWantedAxesOrderAddOnes(shapeInt,
				inputTensorSpec.getAxesOrder(), 
				processingAxesOrder);
        int[] patchGridSize = new int[] {1, 1, 1, 1, 1};
        if (descriptor.isTilingAllowed()) {
            patchGridSize = IntStream.range(0, inputPatchSize.length)
                    .map(i -> (int) Math.ceil((double) inputSequenceSize[i] / ((double) inputPatchSize[i] - halo[i] * 2)))
                    .toArray();
        }
        // For the cases when the patch is bigger than the  image size, share the
        // padding between both sides of the image
        paddingSize[0] = IntStream.range(0, inputPatchSize.length)
                .map(i -> 
                	(int) Math.max(paddingSize[0][i],
                			Math.ceil( (double) (inputPatchSize[i] - inputSequenceSize[i]) / 2))
                ).toArray();
        paddingSize[1] = IntStream.range(0, inputPatchSize.length)
            .map(i -> (int) Math.max( paddingSize[1][i], 
            		inputPatchSize[i] - inputSequenceSize[i] - paddingSize[0][i])).toArray();

        return PatchSpec.create(inputTensorSpec.getName(), inputPatchSize, patchGridSize, paddingSize);
    }
    
    /**
     * Convert the Icy Sequence int[] into another int[] which follows the axes order
     * of a tensor of interest
     * @param seqSize
     * 	icy sequence size array
     * @param axes
     * 	axes order of the tensor of interest
     * @return a size array in the order of the tensor of interest
     */
    public static int[] icySeqAxesOrderToWantedOrder(int[] seqSize, String axes) {
    	String icyAxesOrder = "xyzbc".toUpperCase();
    	return arrayToWantedAxesOrderAddOnes(seqSize, icyAxesOrder, axes);
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
