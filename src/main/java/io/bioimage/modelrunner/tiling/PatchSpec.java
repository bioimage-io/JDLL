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

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Patch specification providing information about the patch size and patch grid size.
 * 
 * @author Daniel Felipe Gonzalez Obando and Carlos Garcia Lopez de Haro
 */
public class PatchSpec
{
	/**
	 * Size of the tensor that is going to be tiled
	 */
    private long[] tensorDims;
	/**
	 * Size of the input patch. Following "xyczb" axes order
	 */
    private long[] patchInputSize;
    /**
     * Size of the number of patches per axis. Following "xyczb" axes order
     */
    private int[] patchGridSize;
    /**
     * Size of the padding for the tensor. The first row represents the padding
     * the left or upper side, and the second on the bottom or right. 
     * Following "xyczb" axes order on each row
     */
    private int[][] patchPaddingSize;
    /**
     * Name of the tensor to whom these specs correspond
     */
    private String tensorName;

    /**
     * Creates a patch size specification.
     * 
      * @param tensorName
     *        Name of the tensor .
     * @param patchInputSize
     *        Input patch size. The patch taken from the input sequence including the halo.
     * @param patchGridSize
     *        The patch grid size. The number of patches in each axis used to cover the entire sequence. It should be computed from the output patch and input
     *        sequence sizes.
     * @param patchPaddingSize
     *        The padding size used on each patch.
    * @return The create patch specification.
     */
    public static PatchSpec create(String tensorName, long[] patchInputSize, int[] patchGridSize, 
    		int[][] patchPaddingSize, long[] tensorDims)
    {
        PatchSpec ps = new PatchSpec();
        ps.patchInputSize = patchInputSize;
        ps.patchGridSize = patchGridSize;
        ps.patchPaddingSize = patchPaddingSize;
        ps.tensorName = tensorName;
        ps.tensorDims = tensorDims;
        return ps;
    }

    private PatchSpec()
    {
    }
    
    /**
     * TODO this method should be per image, not in total??
     * TODO this method should be per image, not in total??
     * TODO this method should be per image, not in total??
     * TODO this method should be per image, not in total??
     * Obtain the number of patches in each axes for a list of input patch specs.
     * When tiling is allowed, only one patch grid is permitted. If among the tensors
     * there are one or more that do not allow tiling, then two patch sizes are allowed,
     * the one for the tensors that allow tiling and the one for the ones that not (that will
     * just be 1s in every axes).
     * In the case there exist tensors that allow tiling, the grid size for those will be the
     * one returned
     * @param patches
     * 	list of specs for the tiling strategy of a list of tensors
     * @return the number of patches in each axes
     */
    public static int[] getGridSize(List<PatchSpec> patches) {
    	// The minimum possible grid is just one patch in every direction. This is the
    	// grid if no tiling is allowed
    	int[] grid = new int[]{1, 1, 1, 1, 1};
    	// If there is any different grid, that will be the absolute one
    	for (PatchSpec pp : patches) {
    		if (!PatchGridCalculator.compareTwoArrays(grid, pp.getPatchGridSize()))
    			return pp.getPatchGridSize();
    	}
    	return grid;
    }
    
    /**
     * TODO this method should be per image, not in total??
     * TODO this method should be per image, not in total??
     * TODO this method should be per image, not in total??
     * TODO this method should be per image, not in total??
     * Obtain the number of patches in each axes for a list of input patch specs.
     * When tiling is allowed, only one patch grid is permitted. If among the tensors
     * there are one or more that do not allow tiling, then two patch sizes are allowed,
     * the one for the tensors that allow tiling and the one for the ones that not (that will
     * just be 1s in every axes).
     * In the case there exist tensors that allow tiling, the grid size for those will be the
     * one returned
     * @param patches
     * 	map containing tiling specs per tensor
     * @return the number of patches in each axes
     */
    public static int[] getGridSize(Map<String, PatchSpec> patches) {
    	// The minimum possible grid is just one patch in every direction. This is the
    	// grid if no tiling is allowed
    	int[] grid = new int[]{1, 1, 1, 1, 1};
    	// If there is any different grid, that will be the absolute one
    	for (PatchSpec pp : patches.values()) {
    		if (!PatchGridCalculator.compareTwoArrays(grid, pp.getPatchGridSize()))
    			return pp.getPatchGridSize();
    	}
    	return grid;
    }
    
    /**
     * Return the PatchSpec corresponding to the tensor called by the name defined
     * @param specs
     * 	list of patch specs
     * @param name
     * 	name of the tensor of interest
     * @return the patch specs of the tensor if interest
     */
    public static PatchSpec getPatchSpecFromListByName(List<PatchSpec> specs, String name) {
    	return specs.stream().filter(pp -> pp.getTensorName().equals(name)).findAny().orElse(null);
    }
    
    /**
     * GEt the name of the tensor
     * @return the name of the tensor
     */
    public String getTensorName() {
    	return tensorName;
    }
    
    /**
     * The dimensions of the tensor
     * @return the dimensions of the tensor that is going to be tiled
     */
    public long[] getTensorDims() {
    	return tensorDims;
    }

    /**
     * @return Input patch size. The patch taken from the input sequence including the halo.
     */
    public long[] getPatchInputSize()
    {
        return patchInputSize;
    }

    /**
     * @return The patch grid size. The number of patches in each axis used to cover the entire sequence. It should be computed from the output patch and input
     *         sequence sizes.
     */
    public int[] getPatchGridSize()
    {
        return patchGridSize;
    }

    /**
     * @return The padding size used on each patch.
     */
    public int[][] getPatchPaddingSize()
    {
        return patchPaddingSize;
    }

    @Override
    public String toString()
    {
    	String[] paddingStrArr = new String[patchPaddingSize[0].length];
    	for (int i = 0; i < paddingStrArr.length; i ++)
    		paddingStrArr[i] = patchPaddingSize[0][i] + "," + patchPaddingSize[1][i];
        StringBuilder builder = new StringBuilder();
        builder.append("PatchSpec of '" + tensorName + "'"
        		+ "[patchInputSize=").append(Arrays.toString(patchInputSize))
                .append(", patchGridSize=").append(Arrays.toString(patchGridSize))
                .append(", patchPaddingSize=").append(Arrays.toString(paddingStrArr))
                .append("]");
        return builder.toString();
    }

}
