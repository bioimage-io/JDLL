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
    private long[] nonTiledTensorDims;
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
     * @param nonTiledTensorDims
     * 		  The original size of the image/tensor that is going to be tiled
     * @return The create patch specification.
     */
    public static PatchSpec create(String tensorName, long[] patchInputSize, int[] patchGridSize, 
    		int[][] patchPaddingSize, long[] nonTiledTensorDims)
    {
        PatchSpec ps = new PatchSpec();
        ps.patchInputSize = patchInputSize;
        ps.patchGridSize = patchGridSize;
        ps.patchPaddingSize = patchPaddingSize;
        ps.tensorName = tensorName;
        ps.nonTiledTensorDims = nonTiledTensorDims;
        return ps;
    }

    private PatchSpec()
    {
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
    public long[] getNonTiledTensorDims() {
    	return nonTiledTensorDims;
    }

    /**
     * @return Input patch size. The patch taken from the input sequence including the halo.
     */
    public long[] getTileSize()
    {
        return patchInputSize;
    }

    /**
     * @return The patch grid size. The number of patches in each axis used to cover the entire sequence. It should be computed from the output patch and input
     *         sequence sizes.
     */
    public int[] getTileGrid()
    {
        return patchGridSize;
    }

    /**
     * @return The padding size used on each patch.
     */
    public int[][] getPadding()
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
