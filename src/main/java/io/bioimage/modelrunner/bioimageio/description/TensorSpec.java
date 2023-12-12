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
package io.bioimage.modelrunner.bioimageio.description;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import io.bioimage.modelrunner.bioimageio.description.exceptions.ModelSpecsException;
import io.bioimage.modelrunner.tiling.PatchGridCalculator;
import io.bioimage.modelrunner.utils.YAMLUtils;


/**
 * A tensor specification descriptor. It holds the information of an input or output tensor (name, shape, axis order, data type, halo, etc.).
 * It is built from a input or output tensor map element in the yaml file.
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public class TensorSpec {
	/**
	 * Whether the tensor represents an input or an output
	 */
    private boolean input;
    /**
     * The name of the tensor
     */
    private String name;
    /**
     * The String axes order of the tensor (Ex: bcyx, xyc...)
     */
    private String axes;
    /**
     * The data type of the tensor
     */
    private String dataType;
    /**
     * The description of the tensor
     */
    private String description;
    /**
     * For inputs, the total padding that the outputs have (halo + offset) and for outputs
     * the halo of a particualr tensor.
     * REgard that the input halo follows the "xyczb" axes order
     * whereas the output follows the tensor axes order.
     * This is done because for several inputs the axes order might change
     * for each of them
     */
    private float[] halo;
    /**
     * The shape information of a tensor
     */
    private ShapeSpec shape;
    /**
     * The type of tensor (image, list)
     */
    private String type;
    /**
     * The list of pre-processing routines
     */
    private List<TransformSpec> preprocessing;
    /**
     * The list of post-processing routines
     */
    private List<TransformSpec> postprocessing;
    /**
     * Whether this tensor allows tiling or not.
     * Tiling is always allowed unless the contrary is defined
     */
    private boolean tiling  = true;
    /**
     * Tile/patch size selected by the user and used to process the image
     */
    private int[] processingTile;
    /**
     * String representing the object tensor type image
     */
    public static final String IMAGE = "image";
    /**
     * String representing the object tensor type list (a matrix of 1 or 2 dimensions).
     * In order for the object to be a list, the shape has to contain either only 1 dimension ("c"),
     * contain 2, one of them being the batch_size ("bc"), or contain the letter "i", ("bic").
     * "I" comes from instance
     */
    public static final String LIST = "list";

    /**
     * Builds the tensor specification instance from the tensor map and an input flag.
     * 
     * @param tensorSpecMap
     *        The map of elements describing the tensor.
     * @param input
     *        Whether it is an input (true) or an output (false) tensor.
     * @return The tensor specification instance.
     * @throws ModelSpecsException if any of the fields does not fulfill the requirements
     */
    @SuppressWarnings("unchecked")
    public static TensorSpec build(Map<String, Object> tensorSpecMap, boolean input) throws ModelSpecsException
    {
        TensorSpec tensor = new TensorSpec();
        tensor.name = (String) tensorSpecMap.get("name");
        tensor.axes = (String) tensorSpecMap.get("axes");
        tensor.dataType = (String) tensorSpecMap.get("data_type");
        tensor.description = (String) tensorSpecMap.get("description");
        tensor.input = input;
        // TODO
        // List<String> rangeList = (List<String>) tensorSpecMap.get("data_range");
        // tensor.range = rangeList == null ? null : new ArrayList<>(rangeList);
        List<?> haloList = (List<?>) tensorSpecMap.get("halo");
        tensor.halo = (input
            ? null
            : (haloList == null ? new float[tensor.axes.length()] : YAMLUtils.castListToFloatArray(haloList)));
        tensor.shape = ShapeSpec.build(tensorSpecMap.get("shape"), input);
        tensor.type = IMAGE;
        if ((tensor.axes == null) ||
            (tensor.axes.length() <= 2 && tensor.axes.toUpperCase().matches(".*[B|I].*"))
            || tensor.axes.toUpperCase().contains("I")|| tensor.axes.length() == 1)
        {
            tensor.type = LIST;
        }
        tensor.processingTile = tensor.shape.getTileRecomendedSize();

        List<?> preprocessingTensors = (List<?>) tensorSpecMap.get("preprocessing");
        if (preprocessingTensors == null)
        {
            tensor.preprocessing = new ArrayList<>(0);
        }
        else
        {
            tensor.preprocessing = new ArrayList<TransformSpec>(preprocessingTensors.size());
            for (Object elem : preprocessingTensors)
            {
                tensor.preprocessing.add(TransformSpec.build((Map<String, Object>) elem));
            }
        }

        List<?> postprocessingTensors = (List<?>) tensorSpecMap.get("postprocessing");
        if (postprocessingTensors == null)
        {
            tensor.postprocessing = new ArrayList<>(0);
        }
        else
        {
            tensor.postprocessing = new ArrayList<TransformSpec>(postprocessingTensors.size());
            for (Object elem : postprocessingTensors)
            {
                tensor.postprocessing.add(TransformSpec.build((Map<String, Object>) elem));
            }
        }

        return tensor;
    }
    
    /**
     * In order to handle tensors that might be too big, or to handle halo to remove border artifacts,
     * the image tensors can be tiled, into smaller images and the passed to the corresponding Neural NEtwork.
     * This method allows setting the size of the tile. Note that the tile size should fulfil the parameters
     * specified in the bioimage.io rdf.yaml file, and if it does not, running the model will fail
     * @param tileSize
     * 	size of the tile the input is going to be divided in
     */
    public void setTileSize(int[] tileSize) {
    	this.processingTile = tileSize;
    }
    
    /**
     * REturn the axes order in a readable format for humans
     * Example: byxc converts into Y,X,C; xbyc converts into X,Y,C
     * @return the axes order in a readable format
     */
    public String getDisplayableAxesOrder() {
    	String displayableAxes = axes.toUpperCase().replace("B", "");
    	String[] displayableAxesArr = displayableAxes.split("");
    	// REturn substring without the first and last chars. Foe ex: [X,Y,C] -> X,Y,C
    	String outStr = Arrays.toString(displayableAxesArr);
    	return outStr.substring(1, outStr.length() - 1);
    }
    
    /**
     * REturn an array of sizes separated by commas and without the batch size
     * Example: if byxc and [1,256,256,3], then 256,256,3
     * @param arr
     * 	arra contianing the whole size of a tensor
     * @return the tile/patch size in displayable format
     */
    public int[] getDisplayableSizes(int[] arr) {
    	int bInd = axes.toUpperCase().indexOf("B");
    	if (bInd == -1) {
    		return arr;
    	}
    	return IntStream.range(0, arr.length)
        .filter(i -> i != bInd)
        .map(i -> arr[i])
        .toArray();
    }
    
    /**
     * REturn a String of an array of sizes separated by commas and without the batch size
     * Example: if byxc and [1,256,256,3], then 256,256,3
     * @param arr
     * 	arra contianing the whole size of a tensor
     * @return the tile/patch size in a String displayable (without b dimension) format
     */
    public String getDisplayableSizesString(int[] arr) {
    	int[] displayableArr = getDisplayableSizes(arr);
    	// In case -1 is shown, display auto to be more human uunderstandable
    	String patchStr = Arrays.toString(displayableArr).replace("-1", "auto");
    	// Remove the begining and end "[" "]"
    	patchStr = patchStr.substring(1,  patchStr.length() - 1);
    	return patchStr;
    }

    /**
     * Set the constraints for each of the axes regarding step and minimum size
     * @return the String containing the info for the tensor corresponding to min and step
     *  in the format: Minimum size: X=8, Y=8, C=0, Z=4   Step: X=8, Y=8, C=1, Z=4
     */
    public String getDisplayableStepMinConstraints() {
    	String minStr = "MINIMUM SIZE: ";
    	String stepStr = "STEP: ";
    	String[] axesArr = axes.toUpperCase().split("");
    	for (int i = 0; i < axesArr.length; i ++) {
    		if (axesArr[i].equals("B"))
    			continue;
    		minStr += axesArr[i] + "=" + shape.getTileMinimumSize()[i] + ", ";
    		stepStr += axesArr[i] + "=" + shape.getTileStep()[i] + ", ";
    	}
    	// Remove the final ", "
    	minStr = minStr.substring(0, minStr.length() - ", ".length());
    	stepStr = stepStr.substring(0, stepStr.length() - ", ".length());
    	return minStr + "    " + stepStr;
    }
    
    /** TODO this method does not realy required to be non-static because the instance already
     * contains the axes order, maybe separate two methods
     * REturn an array containing the optimal patch for the given sequence.
     * @param seqSize
     * 	array containing the size of the sequence with the axes order defined
     * @param seqSizeAxes
     * 	axes order of the size array
     * @return the array with the patch of interest
     */
    public int[] getOptimalTileSize(int[] seqSize, String seqSizeAxes) {
    	return getOptimalTileSizeForTiling(seqSize, seqSizeAxes, tiling);
    }
    
    /**
     * REturn an array containing the optimal patch for the given sequence.
     * TODO add: An optimal patch has the arbitrary requirement of not being bigger than
     * 10^6 pixels
     * @param seqSize
     * 	array containing the size of the sequence with the axes order defined
     * @param seqSizeAxes
     * 	axes order of the size array
     * @param applyTiling
     * 	whether tiling is considered or not.
     * @return the array with the patch of interest
     */
    public int[] getOptimalTileSizeForTiling(int[] seqSize, String seqSizeAxes, boolean applyTiling) {
    	int[] patch = new int[axes.length()];
    	String seqSizeAxesUpper = seqSizeAxes.toUpperCase();
    	String[] axesArr = axes.toUpperCase().split("");
		for (int ii = 0; ii < axesArr.length; ii ++) {
			float haloVal = halo[ii];
			int min = shape.getTileMinimumSize()[ii];
			int step = shape.getTileStep()[ii];
			int ind = seqSizeAxesUpper.indexOf(axesArr[ii]);
			int size = seqSize[ind];
			if (step != 0 && applyTiling) {
				patch[ii] = (int)Math.ceil((size + 2 * haloVal) / step) * step;
				// The patch cannot be 3 times bigger than the size of the image, because then
				// mirroring would stop working
				if (patch[ii] > 3 * size && (patch[ii] - step) >= min) 
					patch[ii] = patch[ii] - step;
				if (patch[ii] > 3 * size && ((int)Math.ceil((double)size / (double)step) * step) >= min) 
					patch[ii] = (int)Math.ceil((double)size / (double)step) * step;
			} else if (step != 0 && !applyTiling){
				// If tiling is not allowed and the step can vary, the patch size for that
				// dimension is left to -1 and then calculated automatically after pre-processing
				// just in case pre-processing changes the image size
				patch[ii] = -1;
			} else if (step == 0){
				patch[ii] = min;
			}
		}
		return patch;
    }
    
    /**
     * REturn an array containing the optimal patch for the given sequence for a tensor that 
     * does not allow tiling. Should only be calculate after pre-processing and right 
     * before feeding the tensor to the model
     * @param seqSize
     * 	array containing the size of the sequence with the axes order defined
     * @param seqSizeAxes
     * 	axes order of the size array
     * @return the array with the patch of interest
     */
    public int[] getOptimalTileSizeNoTiling(int[] seqSize, String seqSizeAxes) {
		return getOptimalTileSizeForTiling(seqSize, seqSizeAxes, true);
    }
    
    /**
     * Gets the patch size as an int[] from a String introduced by the user.
     * The String has the form of tile_dimension1,tile_dimension2,tile_dimension3,...
     * REgards that batch size is ignored by the user, thus it is always set to one
     * @param patchStr
     * 	the patch size as a String introduced by the user
     * @return the patch int array
     */
    public int[] getTileSizeFromStr(String patchStr) {
    	String[] axesArr = axes.toUpperCase().split("");
    	int[] patchArr = new int[axesArr.length];
    	String[] patchstrArr = patchStr.split(",");
    	int count = 0;
    	for (int i = 0; i < axesArr.length; i ++) {
    		if (axesArr[i].equals("B")) {
    			patchArr[i] = 1;
    		} else {
    			patchArr[i] = patchstrArr[count].trim().equals("auto") ? -1 :Integer.parseInt(patchstrArr[count].trim());
    			count ++;
    		}
    	}
    	return patchArr;
    }
    
    /**
     * Validates if a given patch array fulfills the conditions specified in the yaml file.
     * To check whether the tile size works also for a given image, 
     * please use {@link #setTileSizeForTensorAndImageSize(int[], int[])}
     * @param tileSize
     * 	the patch array size to validate, it has to be in the axes order of the tensor
     * @throws Exception if the patch does not comply with the constraints specified
     */
    public void validate(int[] tileSize) throws Exception {
    	// VAlidate that the minimum size and step constraints are fulfilled
    	validateStepMin(tileSize);
    }
    
    /**
     * Validates if a given patch array fulfills the conditions specified in the yaml file.
     * To check whether the tile size works also for a given image, 
     * please use {@link #setTileSizeForTensorAndImageSize(int[], int[])}
     * @param tileSize
     * 	the patch array size to validate, it has to be in the axes order of the tensor
     * @throws Exception if the patch does not comply with the constraints specified
     */
    public void validate(int[] tileSize, String tileAxesOrder) throws Exception {
    	tileSize = PatchGridCalculator.arrayToWantedAxesOrderAddOnes(tileSize, tileAxesOrder, this.axes);
    	// VAlidate that the minimum size and step constraints are fulfilled
    	validateStepMin(tileSize);
    }
    
    /**
     * Sets the tile size to process the given tensor regarding that the tensor
     * has the dimensions specified by the second argument.
     * If also validates if the tile size selected fufils the requirements
     * specified in the bioimage.io rdf.yaml file.
     * Both arguments must follow the same axis order of this tensor.
     * If everything is correct, sets {@link #processingTile} to the first argument.
     * @param tileSize
     * 	the size of the tile/patch in which the main tensor is going to be divided.
     * 	It should follow the tensor axes order as defined in the Bioimage.io rdf.yaml file
     * @param tensorImageSize
     * 	size of the image that is used for the tensor.
     * 	It should follow the tensor axes order as defined in the Bioimage.io rdf.yaml file
     * @throws Exception if the patch size is not able to 
     * 	fulfill the requirements of the tensor
     */
    public void setTileSizeForTensorAndImageSize(int[] tileSize, int[] tensorImageSize) throws Exception {
    	if (tileSize.length != tensorImageSize.length)
    		throw new Exception("tileSize and tensorImageSize need to have the same number of dimensions.");
    	// If tiling is not allowed, the patch array needs to be equal to the
    	// optimal patch
    	if (!tiling) {
    		validateNoTiling(tileSize, tensorImageSize);
    	}
    	// VAlidate that the minimum size and step constraints are fulfilled
    	validateStepMin(tileSize);
    	// Finally validate that the sequence size complies with the patch size selected
    	validateTileSizeVsImage(tileSize, tensorImageSize);
    	this.processingTile = tileSize;
    }
    
    /**
     * Sets the tile size to process the given tensor regarding that the tensor
     * has the dimensions specified by the second argument.
     * If also validates if the tile size selected fufils the requirements
     * specified in the bioimage.io rdf.yaml file.
     * Both arguments must follow the same axis order of this tensor.
     * If everything is correct, sets {@link #processingTile} to the first argument.
     * @param tileSize
     * 	the size of the tile/patch in which the main tensor is going to be divided
     * @param tileAxesOrder
     * 	axes order of the tile provided
     * @param tensorImageSize
     * 	size of the image that is used for the tensor.
     * @throws Exception if the patch size is not able to 
     * 	fulfill the requirements of the tensor
     */
    public void setTileSizeForTensorAndImageSize(int[] tileSize, String tileAxesOrder,
    		int[] tensorImageSize, String imageAxesOrder) throws Exception {
    	if (tileSize.length != tileAxesOrder.length())
    		throw new Exception("tileSize and tileAxesOrder should have the same length.");
    	if (tensorImageSize.length != imageAxesOrder.length())
    		throw new Exception("tensorImageSize and imageAxesOrder should have the same length.");
    	tileSize = PatchGridCalculator.arrayToWantedAxesOrderAddOnes(tileSize, tileAxesOrder, this.axes);
    	tensorImageSize = PatchGridCalculator.arrayToWantedAxesOrderAddOnes(tensorImageSize, imageAxesOrder, this.axes);
    	// If tiling is not allowed, the patch array needs to be equal to the
    	// optimal patch
    	if (!tiling) {
    		validateNoTiling(tileSize, tensorImageSize);
    	}
    	// VAlidate that the minimum size and step constraints are fulfilled
    	validateStepMin(tileSize);
    	// Finally validate that the sequence size complies with the patch size selected
    	validateTileSizeVsImage(tileSize, tensorImageSize);
    	this.processingTile = tileSize;
    }

    
    /**
     * VAlidate that the wanted tile size, is compatible with the image size.
     * The patch cannot be 3 times bigger than the image size because mirroring
     * would not work, cannot be smaller than total halo * 2, and patching cannot
     * exist along the channels dimension
     * @param tileSize
     * 	the proposed tile/patch size
     * @param seqSize
     * 	the sequence size
     * @throws Exception if any of the constraints is not fulfilled
     */
    private void validateTileSizeVsImage(int[] tileSize, int[] seqSize) throws Exception {
    	validateTileVsImageSize(tileSize, seqSize);
    	validateTileVsHalo(tileSize);
    	validateTileVsImageChannel(tileSize, seqSize);
    }
    
    /**
     * VAlidate that the tile/patch selected, is compatible with the image size.
     * The tile/patch cannot be 3 times bigger than the image size because mirroring
     * would not work.
     * @param tileSize
     * 	the proposed tile/patch size
     * @param seqSize
     * 	the sequence size
     * @throws Exception if any of the constraints is not fulfilled
     */
    private void validateTileVsImageSize(int[] tileSize, int[] seqSize) throws Exception {
    	boolean tooBig = IntStream.range(0, tileSize.length).anyMatch(i -> tileSize[i] > seqSize[i] * 3);
    	if (tooBig) {
    		int[] maxPatch = new int[seqSize.length];
    		IntStream.range(0, tileSize.length).forEach(i -> maxPatch[i] = seqSize[i] * 3);
    		throw new Exception("Error in the axes size selected.\n"
    						+ "The axes size introduced in any of the dimensions cannot\n"
    						+ "be bigger than 3 times the Sequence size.\n"
    						+ "The Sequence selected for tensor '" + name + "' has\n"
							+ "the following dimensions: " + getDisplayableSizesString(seqSize) 
							+ " for axes " + getDisplayableAxesOrder() + ".\n"
							+ "With those dimensions the biggest axes size \n"
							+ "for each dimension should be: " + getDisplayableSizesString(maxPatch) + ".\n"
							+ "However, the axes size introduced is: " + getDisplayableSizesString(tileSize) + ".");
    	}
    }
    
    /**
     * VAlidate that the tile/patch selected, is compatible with the image size.
     * The patch cannot be smaller than total halo * 2.
     * @param tile
     * 	the proposed patch size
     * @param seqSize
     * 	the sequence size
     * @throws Exception if any of the constraints is not fulfilled
     */
    private void validateTileVsHalo(int[] tile) throws Exception {
    	boolean tooSmall = IntStream.range(0, tile.length).anyMatch(i -> tile[i] <= halo[i] * 2);
    	if (tooSmall) {
    		int[] minPatch = new int[halo.length];
    		for (int i = 0; i < halo.length; i ++) {minPatch[i] = (int) (halo[i] * 2);}
    		throw new Exception("Error in the axes size selected.\n"
    						+ "The axes size introduced in any of the dimensions cannot\n"
    						+ "be smaller than 2 times the total halo size.\n"
    						+ "Regarding the total halo (max(offset + halo)) of"
    						+ "tensor '" + name + "' the minimum size for each \n"
							+ "dimension should be: " + getDisplayableSizesString(minPatch) + ".\n"
							+ "However, the axes size introduced is: " + getDisplayableSizesString(tile) 
							+ " for axes " + getDisplayableAxesOrder() + ".");
    	}
    }
    
    /**
     * VAlidate that the patch selected, is compatible with the image size.
     * The patch cannot be different than the image size along the channel dimension
     * @param tile
     * 	the proposed patch size
     * @param seqSize
     * 	the sequence size
     * @throws Exception if any of the constraints is not fulfilled
     */
    private void validateTileVsImageChannel(int[] tile, int[] seqSize) throws Exception {
    	int channelInd = axes.toLowerCase().indexOf("c");
    	if (channelInd != -1 && tile[channelInd] != seqSize[channelInd]) {
    		throw new Exception("Error in the axes size selected.\n"
					+ "DeepIcy does not allow tiling along the channels dimension.\n"
					+ "The axes size introduced for axis 'C' should be equal to\n"
					+ "the number of channels in the image.\n"
					+ "For input tensor '" + name + "', the sequence selected\n"
					+ "has '" + seqSize[channelInd] + "' channels whereas the\n"
					+ "axes size for 'C' is '" + tile[channelInd] + "'.");
    	}
    }
    
    /**
     * Validate that the  patch size is a product of min_size + step * n where n can be any integer >= 0
     * @param tileSize
     * 	the size of the tile/patch introduced by the user
     * @throws Exception if the patch does not fulfill the min and step conditions
     */
    private void validateStepMin(int[] tileSize) throws Exception {
    	boolean badSize = IntStream.range(0, tileSize.length)
				.anyMatch(i -> 
				shape.getTileStep()[i] != 0 ?
						(tileSize[i] - shape.getTileMinimumSize()[i]) % shape.getTileStep()[i] != 0
						: tileSize[i] != shape.getTileMinimumSize()[i]
				);
    	if (badSize) {
    		throw new Exception("Error in the axes size selected.\n"
    				+ "Tensor " + getName() + " with the following requirements:\n"
    				+ getDisplayableStepMinConstraints() + " are not compatible with\n"
					+ "the introduced tile size (" + getDisplayableSizesString(tileSize) + ").\n"
					+ "Regard that when step = 0, the tile size has to be equal\n"
					+ "to the minimum size");
    	}
    }
    
    /**
     * Validate that the size of the patch is correct in the case that the tensor
     * does not allow tiling
     * @param tile
     * 	size of the patch following the axes order
     * @param seqSize
     * 	size of the image following the tensor axes order
     * @throws Exception if there is any issue with the validation
     */
    private void validateNoTiling(int[] tile, int[] seqSize) throws Exception {
    	int[] optimalPatch = getOptimalTileSize(seqSize, axes);
    	boolean patchEqualsOptimal = true;
    	for (int i = 0; i < tile.length; i ++) {
    		if (optimalPatch[i] != tile[i]) {
    			patchEqualsOptimal = false;
    			break;
    		}
    	}
    	boolean seqBiggerThanPatch = true;
    	for (int i = 0; i < tile.length; i ++) {
    		if (seqSize[i] > tile[i]) {
    			seqBiggerThanPatch = false;
    			break;
    		}
    	}
    	if (!patchEqualsOptimal && !seqBiggerThanPatch){
    		throw new Exception(" Error in the axes size selected.\n"
    				+ "Tensor " + getName() + " does not allow tiling and due to the tensor specs\n"
					+ "and image size (" + Arrays.toString(getDisplayableSizes(seqSize)) + ") the tile size can\n"
					+ "only be (" + getDisplayableSizes(optimalPatch) + ") in order to process the\n"
					+ "whole image at once. These dimensions do not coincide with the ones\n"
					+ "introduced (" + Arrays.toString(getDisplayableSizes(tile)) + ").");
    	}
    	if (seqBiggerThanPatch){
    		throw new Exception("Error in the tiling size selected. "
    				+ "Tensor " + getName() + " does not allow tiling and the "
					+ "image size (" + Arrays.toString(getDisplayableSizes(seqSize)) + ") is bigger than the"
					+ " tile size (" + Arrays.toString(getDisplayableSizes(tile)) + ") selected. "
					+ "With this parameters it would be impossible to process the whole"
					+ "image without tiling.");
    	}
    	this.processingTile = getOptimalTileSizeNoTiling(seqSize, axes);
    }
    
    /**
     * Sets the total halo for the inputs defined by the outputs.
     * REgard that the input halo follows the tensor axes order
     * @param totalHalo
     * 	total halo, amount of pixels that have to be removed from the sides of each dim
     */
    public void setTotalHalo(float[] totalHalo){
    	this.halo = new float[axes.length()];
    	String totalHaloAxesOrder = "XYCZB".toLowerCase();
    	String[] axesArr = axes.toLowerCase().split("");
    	for (int i = 0; i < axesArr.length; i ++) {
    		int ind = totalHaloAxesOrder.indexOf(axesArr[i]);
    		this.halo[i] = totalHalo[ind];
    	}
    }
    
    /**
     * Return a String containing the dimensions of the optimal
     * tile/patch for the selected image
     * @param imSize
     * 	size of the image. The order is Width, Height, Channel, Slices, time
     * @param axesOrder
     * 	the axes order of the imSize parameter
     * @return the String containing a tile/patch in the format 256,256,3
     */
    public String getDisplayableOptimalTile(int[] imSize, String axesOrder) {
    	// Remove the B axes from the optimal patch size and get the String representation
    	String patchStr = getDisplayableSizesString(getOptimalTileSize(imSize, axesOrder));
    	return patchStr;
    }
    
    /**
     * Validates the selected tile/patch array fulfills the conditions specified in the yaml file
     * with respect to the tensor size before introducing it into the model
     * If it is valid, it sets the value as the {@link #processingTile}
     * @param seqSize
     * 	size of the tensor before inference (after pre-processing)
     * @param axesOrder
     * 	axes order of the tensor
     * @throws Exception  if the seqsize is not compatible with the the tensor constraints
     */
    public void validateTensorSize(int[] seqSize, String axesOrder) throws Exception {
    	// Convert the Icy sequence array dims into the tensor axes order
    	seqSize = PatchGridCalculator.arrayToWantedAxesOrderAddOnes(seqSize, axesOrder, axes);
    	// If tiling is not allowed, the patch array needs to be equal to the
    	// optimal patch
    	if (!tiling) {
    		validateNoTiling(processingTile, seqSize);
    	}
    	// VAlidate that the minimum size and step constraints are fulfilled
    	validateStepMin(processingTile);
    	// Finally validate that the sequence size complies with the patch size selected
    	validateTileSizeVsImage(processingTile, seqSize);
    }


    /**
     * REturn the tile/patch size for this tensor introduced by the user for processing
     * @return the tile size to process the tensor
     */
    public int[] getTileSize() {
    	return this.processingTile;
    }

    /**
     * Return whether the tensor represents an image or not.
     * Currently only the types {@link #IMAGE} and {@link #LIST}
     * are supported. 
     * In order for the object to be a list, the shape has to contain either only 1 dimension ("c"),
     * contain 2, one of them being the batch_size ("bc"), or contain the letter "i", ("bic").
     * "I" comes from instance
     * An image is everythin else.
     * 
     * @return The type of tensor. As of now it can hold "image" or "list" values.
     */
    public String getType()
    {
        return type;
    }

    /**
     * @return The standard preprocessing applied to this tensor. Used when this is an input tensor.
     */
    public List<TransformSpec> getPreprocessing()
    {
        return preprocessing;
    }

    /**
     * @return The standard postprocessing applied to this tensor. Used when this is an output tensor.
     */
    public List<TransformSpec> getPostprocessing()
    {
        return postprocessing;
    }
    
    /**
     * Sets whether tiling is allowed or not
     * @param tiling
     * 	whether tiling is allowed or not
     */
    public void setTiling(boolean tiling) {
    	this.tiling = tiling;
    }
    
    /**
     * Gets whether tiling is allowed or not
     * @return whether tiling is allowed or not
     */
    public boolean getTiling() {
    	return tiling;
    }

    /**
     * @return The data type accepted by the tensor. It can be float32, float64, (u)int8, (u)int16, (u)int32, (u)int64. However, when passing the tensor to the
     *         model it must be of type float.
     */
    public String getDataType()
    {
        return dataType;
    }

    /**
     * @return The halo size on each axis used to crop the output tile.
     *         It is the padding used on each axis at both begin and end on each axis of the image.
     */
    public float[] getHalo()
    {
        return halo;
    }

    /**
     * @return The shape specification of the tensor. It holds either the recommended tensor size or the elements to establish the correct size.
     */
    public ShapeSpec getShape()
    {
        return shape;
    }

    /**
     * @return the description of the tensor.
     */
    public String getDescription()
    {
        return description;
    }

    /**
     * @return True if this instance describes an input tensor. False if it's an output tensor.
     */
    public boolean isInput()
    {
        return input;
    }

    /**
     * @return The name of this tensor. It is usually the name found in the saved model.
     */
    public String getName()
    {
        return name;
    }

    /**
     * A string containing the axis order. Can contain characters "bitczyx". It has no specific size, but it must not be more than 7 and must contain each
     * character at most once.
     * 
     * @return axis order.
     */
    public String getAxesOrder()
    {
        return axes;
    }

    @Override
    public String toString()
    {
        return "TensorSpec {input=" + input + ", name=" + name + ", axes=" + axes + ", dataType=" + dataType + ", halo="
                + halo + ", shape=" + shape + ", type=" + type + ", preprocessing=" + preprocessing
                + ", postprocessing=" + postprocessing + "}";
    }
    
    /**
     * Return whether the tensor represents an image or not.
     * Currently only the types {@link #IMAGE} and {@link #LIST}
     * are supported. 
     * In order for the object to be a list, the shape has to contain either only 1 dimension ("c"),
     * contain 2, one of them being the batch_size ("bc"), or contain the letter "i", ("bic").
     * "I" comes from instance
     * An image is everythin else.
     * 
     * @return whether the tensor represents an image or not
     */
    public boolean isImage() {
    	return this.type.equals(IMAGE);
    }

}
