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

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.Constants;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * A calculator for the size of each patch and the patch grid associated 
 * to input images when applying a given Bioimage.io model.
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando 
 */
public class PatchGridCalculator <T extends RealType<T> & NativeType<T>> 
{
	/**
	 * Instance of {@link ModelDescriptor} containing all the info stored in the specs of an rdf.yaml file
	 */
    private ModelDescriptor descriptor;
    /**
     * MAp containing all the inputs provided to patch, they might be images or not, and might 
     * correspond to the model of interest or not. This class will organize them
     */
    private Map<String, Tensor<T>> inputValuesMap;
    /**
     * MAp containing the {@link PatchSpec} for each of the tensors defined in the rdf.yaml specs file
     */
    private LinkedHashMap<String, PatchSpec> inputTilesSpecs;
    /**
     * MAp containing the {@link PatchSpec} for each of the tensors defined in the rdf.yaml specs file
     */
    private LinkedHashMap<String, PatchSpec> outputTilesSpecs;

    /**
     * Class to calculate the patch specifications given a series of inputs
     * and their specifications. The tiling/patching specs are calculated
     * for both inputs and outputs
     * @param descriptor
     * 	the specifications of each input
     * @param tensorList
     * 	map containing the input images associated to their input tensors
     * @throws IllegalArgumentException if the {@link #inputValuesMap}
     */
    private PatchGridCalculator(ModelDescriptor descriptor, List<Tensor<T>> tensorList)
    		throws IllegalArgumentException
    {
    	for (TensorSpec tt : descriptor.getInputTensors()) {
    		if (tt.isImage() && Tensor.getTensorByNameFromList(tensorList, tt.getName()) == null)
    			throw new IllegalArgumentException("Model input tensor '" + tt.getName() + "' is specified in the rdf.yaml specs file "
    					+ "but cannot be found in the model inputs map provided.");
    		// TODO change isImage() by isTensor()
    		if (tt.isImage() && !(Tensor.getTensorByNameFromList(tensorList, tt.getName()) instanceof Tensor))
    			throw new IllegalArgumentException("Model input tensor '" + tt.getName() + "' is specified in the rdf.yaml specs file "
    					+ "as a tensor but. JDLL needs tensor to be specified either as JDLL tensors (io.bioimage.tensor.Tensor) "
    					+ "or ImgLib2 Imgs (net.imglib2.img.Img), ImgLib2 RandomAccessibleIntervals (net.imglib2.RandomAccessibleInterval) "
    					+ "or ImgLib2 IterableIntervals (net.imglib2.IterableInterval). However, input "
    					+ "'" + tt.getName() + "' is defined as: " + Tensor.getTensorByNameFromList(tensorList, tt.getName()).getClass());
    	}
    	this.descriptor = descriptor;
    	this.inputValuesMap = tensorList.stream().collect(Collectors.toMap(t -> t.getName(), t -> t));
    }
    
    /**
     * Create an instance of {@link PatchGridCalculator} that can be used to calculate
     * how can the inputs to a model be tiled/patched according to the specs in the rdf.yaml file
     *  (only for Bioimage.io models or models that have a Bioimage.io rdf.yaml associated).
     *  
     * @param <T>
     * 	generic type of the possible ImgLibb2 datatypes that input images can have
     * @param modelFolder
     * 	path to the foler of a bioimage.io model. This is the folder where the rdf.yaml
     * 	of the model is located
     * @param inputTensors
     * 	list containing the input tensors
     * @return the object that creates a list of patch specs for each tensor
     * @throws IOException if it is not possible to read the rdf.yaml file of the model or it
     * 	does not exist
     */
    public static <T extends RealType<T> & NativeType<T>> PatchGridCalculator<T> build(String modelFolder, List<Tensor<T>> inputTensors) throws IOException {
    	ModelDescriptor descriptor;
    	try {
	    	descriptor = 
	    			ModelDescriptor.readFromLocalFile(modelFolder + File.separator + Constants.RDF_FNAME, false);
    	} catch (Exception ex) {
    		throw new IOException("Unable to process the rf.yaml specifications file.", ex);
    	}
    	return new PatchGridCalculator<T>(descriptor, inputTensors);
    }
    
    /**
     * Create an instance of {@link PatchGridCalculator} that can be used to calculate
     * how can the inputs to a model be tiled/patched according to the specs in the rdf.yaml file
     *  (only for Bioimage.io models or models that have a Bioimage.io rdf.yaml associated).
     *  
     * @param <T>
     * 	generic type of the possible ImgLibb2 datatypes that input images can have
     * @param model
     * 	model specs as defined in the rdf.yaml file
     * @param inputImagesList
     * 	list of images that correspond to the model inputs specified in the rdf.yaml file.
     * 	The images should be in the same order as the inputs in the rdf.yaml file. First image corresponds
     * 	to the first input, second image to second output and so on.
     * @return the object that creates a list of patch specs for each tensor
     */
    public static <T extends NativeType<T> & RealType<T>> 
    PatchGridCalculator<T> build(ModelDescriptor model, List<Tensor<T>> inputImagesList) {
    	if (inputImagesList.size() != model.getInputTensors().size())
    		throw new IllegalArgumentException("The size of the list containing the model input RandomAccessibleIntervals"
    						+ " was not the same size (" + inputImagesList.size() + ") as the number of "
    						+ "inputs to the model as defined in the rdf.yaml file(" + model.getInputTensors().size() + ").");
    	return new PatchGridCalculator<T>(model, inputImagesList);
     }

    /**
     * Computes the patch size adapted for the input sequence using the model tensor specification.
     * 
     * @return the LinkedHashMap where the key corresponds to the name of the tensor and the value is its
     *  patch specifications
     * @throws IllegalArgumentException if one tensor that allows tiling needs more patches
     * 	in any given axis than the others
     */
    public LinkedHashMap<String, PatchSpec> getInputTensorsTileSpecs() throws IllegalArgumentException
    {
    	if (this.inputTilesSpecs != null)
    		return inputTilesSpecs;
    	List<TensorSpec> inputTensors = findInputImageTensorSpec();
        List<Tensor<T>> inputImages = inputTensors.stream()
        		.filter(k -> this.inputValuesMap.get(k.getName()) != null)
        		.map(k -> this.inputValuesMap.get(k.getName())).collect(Collectors.toList());
        if (inputImages.size() == 0)
        	throw new IllegalArgumentException("No inputs have been provided that match the "
        			+ "specified input tensors specified in the rdf.yaml file.");
        LinkedHashMap<String, PatchSpec> specsMap = computePatchSpecsForEveryTensor(inputTensors, inputImages);
        // Check that the obtained patch specs are not going to cause errors
        checkPatchSpecs(specsMap);
        inputTilesSpecs = specsMap;
        return inputTilesSpecs;
    }
    
    /**
     * Check that the relationship between tensor and image is the same for all the tensors
     * that allow tiling
     * @param patchSpecs
     * 	LinkedHashMap where the key corresponds to the name of the tensor and the value is its
     *  patch specifications
     * @throws IllegalArgumentException if one tensor that allows tiling needs more patches
     * 	in any given axis than the others
     */
    public void checkPatchSpecs(LinkedHashMap<String, PatchSpec> patchSpecs) throws IllegalArgumentException {
    	int[] grid = null;
    	String firstName = null;
    	for (Entry<String, PatchSpec> spec : patchSpecs.entrySet()) {
    		int[] nGrid = spec.getValue().getTileGrid();
    		TensorSpec tt = this.descriptor.findInputTensor(spec.getKey());
    		if (grid == null && tt.getTiling()) {
    			grid = nGrid;
    			firstName = spec.getKey();
    		}
    		if (tt.getTiling() && !compareTwoArrays(nGrid, grid)){
    			throw new IllegalArgumentException("All the input images must be processed with the same number of patches.\n"
						+ "The relationship between the patch size and image size should be the same for every input that allows patching/tiling.\n"
						+ "Tensors '" + firstName + "' and '" + spec.getKey() + "' need different number of patches to "
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
        return this.descriptor.getInputTensors().stream().filter(tr -> tr.isImage())
                .collect(Collectors.toList());
    }

    /**
     * Get the output tensors that correspond to images
     * @return list of tensor specs corresponding to each of the output image tensors
     */
    private List<TensorSpec> findOutputImageTensorSpec()
    {
        return this.descriptor.getOutputTensors().stream().filter(tr -> tr.isImage())
                .collect(Collectors.toList());
    }
    
    /**
     * Create list of patch specifications for every tensor aking into account the
     * corresponding image
     * @param tensors
     * 	the tensor information
     * @param images
     * 	the images corresponding to each tensor
     * @return the LinkedHashMap where the key corresponds to the name of the tensor and the value is its
     *  patch specifications
     */
    private LinkedHashMap<String, PatchSpec> computePatchSpecsForEveryTensor(List<TensorSpec> tensors, List<Tensor<T>> images){
    	LinkedHashMap<String, PatchSpec> patchInfoList = new LinkedHashMap<String, PatchSpec>();
    	for (int i = 0; i < tensors.size(); i ++)
    		patchInfoList.put(tensors.get(i).getName(), computePatchSpecs(tensors.get(i), images.get(i)));
    	return patchInfoList;
    }
    
    /**
     * This method can only be executed if {@link #getInputTensorsTileSpecs()} has already been called,
     * if not it will throw an {@link IllegalArgumentException}.
     * It returns a Map containing instances of {@link PatchSpec} per each of the input image tensors
     * containing the information about the tiles/patches required to process each of the input image tensors.
     * 
     * @return a Map containing instances of {@link PatchSpec} per each of the input image tensors
     * containing the information about the tiles/patches required to process each of the input image tensors.
     * @throws IllegalArgumentException if the input patch specs have not been already calculated
     */
    public LinkedHashMap<String, PatchSpec> getOutputTensorsTileSpecs() throws IllegalArgumentException {
    	if (this.inputTilesSpecs == null)
    		throw new IllegalArgumentException("Please first calculate the tile specs for the input tensors. Call: "
					+ "getInputTensorsTileSpecs()");
    	if (this.outputTilesSpecs != null)
    		return outputTilesSpecs;
    	List<TensorSpec> outTensors = findOutputImageTensorSpec();
    	LinkedHashMap<String, PatchSpec> patchInfoList = new LinkedHashMap<String, PatchSpec>();
		for (int i = 0; i < outTensors.size(); i ++) {
    		String refTensor = outTensors.get(i).getShape().getReferenceInput();
    		PatchSpec refSpec = refTensor == null ? inputTilesSpecs.values().stream().findFirst().get() : inputTilesSpecs.get(refTensor);
    		patchInfoList.put(outTensors.get(i).getName(), computePatchSpecsForOutputTensor(outTensors.get(i), refSpec));
		}
		outputTilesSpecs = patchInfoList;
    	return outputTilesSpecs;
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
    private PatchSpec computePatchSpecs(TensorSpec inputTensorSpec, Tensor<T> inputObject)
    		throws IllegalArgumentException {
    	return computePatchSpecs(inputTensorSpec, inputObject.getData());
    }
    
    private PatchSpec computePatchSpecs(TensorSpec spec, RandomAccessibleInterval<T> rai) throws IllegalArgumentException
    {
    	int[] intShape = new int[rai.dimensionsAsLongArray().length];
    	for (int i = 0; i < intShape.length; i ++) intShape[i] = (int) rai.dimensionsAsLongArray()[i];
    	if (spec.getTileSize() == null) {
			try {
				spec.setTileSizeForTensorAndImageSize(spec.getOptimalTileSize(intShape, spec.getAxesOrder()), intShape);
			} catch (Exception e) {
				throw new IllegalArgumentException("Tensor dimensions of tensor named '" + spec.getName() + "' "
						+ "are not compatible with the requirements set by the"
						+ " rdf.yaml file for tensor '" + spec.getName() + "': " + e.getMessage());
			}
    	}
    	long[] tileSize = Arrays.stream(spec.getTileSize()).mapToLong(i -> i).toArray();
    	return computePatchSpecs(spec, rai, tileSize);
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
    private PatchSpec computePatchSpecs(TensorSpec spec, RandomAccessibleInterval<T> rai, long[] tileSize)
    {
        int[][] paddingSize = new int[2][tileSize.length];
        // REgard that the input halo represents the output halo + offset 
        // and must be divisible by 0.5. 
        float[] halo = spec.getHalo();
        if (!descriptor.isPyramidal() && spec.getTiling()) {
        	// In the case that padding is asymmetrical, the left upper padding has the extra pixel
            for (int i = 0; i < halo.length; i ++) {paddingSize[0][i] = (int) Math.ceil(halo[i]);}
            // In the case that padding is asymmetrical, the right bottom padding has one pixel less
            for (int i = 0; i < halo.length; i ++) {paddingSize[1][i] = (int) Math.floor(halo[i]);}
            
        }
        long[] shapeLong = rai.dimensionsAsLongArray();
        int[] patchGridSize = new int[shapeLong.length];
        for (int i = 0; i < patchGridSize.length; i ++) patchGridSize[i] = 1;
        if (descriptor.isTilingAllowed()) {
            patchGridSize = IntStream.range(0, tileSize.length)
                    .map(i -> (int) Math.ceil((double) shapeLong[i] / ((double) tileSize[i] - halo[i] * 2)))
                    .toArray();
        }
        // For the cases when the patch is bigger than the  image size, share the
        // padding between both sides of the image
        paddingSize[0] = IntStream.range(0, tileSize.length)
                .map(i -> 
                	(int) Math.max(paddingSize[0][i],
                			Math.ceil( (double) (tileSize[i] - shapeLong[i]) / 2))
                ).toArray();
        paddingSize[1] = IntStream.range(0, tileSize.length)
            .map(i -> (int) Math.max( paddingSize[1][i], 
            		tileSize[i] - shapeLong[i] - paddingSize[0][i])).toArray();

        return PatchSpec.create(spec.getName(), tileSize, patchGridSize, paddingSize, rai.dimensionsAsLongArray());
    }
    
    private PatchSpec computePatchSpecsForOutputTensor(TensorSpec tensorSpec, PatchSpec refTilesSpec)
    {
    	int[] inputTileGrid = refTilesSpec.getTileGrid();
    	String ogAxes = ModelDescriptor.findTensorInList(refTilesSpec.getTensorName(), descriptor.getInputTensors()).getAxesOrder();
    	inputTileGrid = arrayToWantedAxesOrderAddOnes(inputTileGrid, ogAxes, tensorSpec.getAxesOrder());
        // REgard that the input halo represents the output halo + offset 
        // and must be divisible by 0.5. 
        int[][] paddingSize = refTilesSpec.getPadding();
        paddingSize[0] = arrayToWantedAxesOrderAddZeros(paddingSize[0], ogAxes, tensorSpec.getAxesOrder());
        paddingSize[1] = arrayToWantedAxesOrderAddZeros(paddingSize[1], ogAxes, tensorSpec.getAxesOrder());
        long[] tileSize;
        long[] shapeLong;
        if (tensorSpec.getShape().getReferenceInput() == null && !tensorSpec.getTiling()) {
        	shapeLong = Arrays.stream(tensorSpec.getTileSize()).mapToLong(i -> i).toArray();
        	tileSize = shapeLong;
        } else if (tensorSpec.getShape().getReferenceInput() == null) {
        	tileSize = Arrays.stream(tensorSpec.getTileSize()).mapToLong(i -> i).toArray();
        	double[] inputTileToTotal = IntStream.range(0, refTilesSpec.getNonTiledTensorDims().length)
        			.mapToDouble(i -> ((double) refTilesSpec.getNonTiledTensorDims()[i]) / ((double) refTilesSpec.getTileSize()[i]))
        			.toArray();
        	float[] floatInputTileToTotal = new float[inputTileToTotal.length];
        	for (int ii = 0; ii < floatInputTileToTotal.length; ii ++) floatInputTileToTotal[ii] = (float) inputTileToTotal[ii];
        	float[] outTileToTotal = arrayToWantedAxesOrderAddOnes(floatInputTileToTotal, ogAxes, tensorSpec.getAxesOrder());
        	shapeLong = IntStream.range(0, tensorSpec.getAxesOrder().length())
        			.mapToLong(i -> (long) Math.ceil(tileSize[i] * outTileToTotal[i])).toArray();
        } else {
        	tileSize = IntStream.range(0, tensorSpec.getAxesOrder().length())
            		.map(i -> (int) (refTilesSpec.getTileSize()[i] * tensorSpec.getShape().getScale()[i] + 2 * tensorSpec.getShape().getOffset()[i]))
            		.mapToLong(i -> i).toArray();
        	shapeLong = LongStream.range(0, tensorSpec.getAxesOrder().length())
            		.map(i -> (int) (refTilesSpec.getNonTiledTensorDims()[(int) i] * tensorSpec.getShape().getScale()[(int) i] 
            				+ 2 * tensorSpec.getShape().getOffset()[(int) i])).toArray();
        }

        return PatchSpec.create(tensorSpec.getName(), tileSize, inputTileGrid, paddingSize, shapeLong);
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
