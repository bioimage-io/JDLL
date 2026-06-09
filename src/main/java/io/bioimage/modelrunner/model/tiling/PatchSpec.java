/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.model.tiling;

import java.util.Arrays;
import java.util.List;

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
     * Axes order followed by the patch grid.
     */
    private String axesOrder;
    /**
     * Axes order used to enumerate flat patch indices.
     */
    private String referenceAxesOrder;
    /**
     * Grid size used to enumerate flat patch indices.
     */
    private int[] referenceTileGrid;

    /**
     * Creates a patch size specification.
     *
     * @param tensorName the tensor name.
     * @param patchInputSize the patch input size.
     * @param patchGridSize the patch grid size.
     * @param patchPaddingSize the patch padding size.
     * @param nonTiledTensorDims the non tiled tensor dimensions.
     * @return the created patch spec.
     */
    protected static PatchSpec create(String tensorName, long[] patchInputSize, int[] patchGridSize,
            int[][] patchPaddingSize, long[] nonTiledTensorDims)
    {
        return create(tensorName, patchInputSize, patchGridSize, patchPaddingSize, nonTiledTensorDims, null, null, null);
    }

    /**
     * Creates a patch size specification.
     *
     * @param tensorName the tensor name.
     * @param patchInputSize the patch input size.
     * @param patchGridSize the patch grid size.
     * @param patchPaddingSize the patch padding size.
     * @param nonTiledTensorDims the non tiled tensor dimensions.
     * @param axesOrder axes order followed by the patch grid.
     * @param referenceAxesOrder axes order used to enumerate flat patch indices.
     * @param referenceTileGrid grid size used to enumerate flat patch indices.
     * @return the created patch spec.
     */
    protected static PatchSpec create(String tensorName, long[] patchInputSize, int[] patchGridSize,
            int[][] patchPaddingSize, long[] nonTiledTensorDims, String axesOrder, String referenceAxesOrder,
            int[] referenceTileGrid)
    {
        PatchSpec ps = new PatchSpec();
        ps.patchInputSize = patchInputSize;
        ps.patchGridSize = patchGridSize;
        ps.patchPaddingSize = patchPaddingSize;
        ps.tensorName = tensorName;
        ps.nonTiledTensorDims = nonTiledTensorDims;
        ps.axesOrder = normalizeAxesOrder(axesOrder);
        ps.referenceAxesOrder = normalizeAxesOrder(referenceAxesOrder);
        ps.referenceTileGrid = referenceTileGrid;
        return ps;
    }

    private PatchSpec()
    {
    }
    
    /**
     * GEt the name of the tensor
     * @return the name of the tensor
     */
    protected String getTensorName() {
    	return tensorName;
    }
    
    /**
     * The dimensions of the tensor
     * @return the dimensions of the tensor that is going to be tiled
     */
    protected long[] getNonTiledTensorDims() {
    	return nonTiledTensorDims;
    }

    /**
     * @return Input patch size. The patch taken from the input sequence including the halo.
     */
    protected long[] getTileSize()
    {
        return patchInputSize;
    }

    /**
     * @return The patch grid size. The number of patches in each axis used to cover the entire sequence. It should be computed from the output patch and input
     *         sequence sizes.
     */
    protected int[] getTileGrid()
    {
        return patchGridSize;
    }

    /**
     * @return The padding size used on each patch.
     */
    protected int[][] getPadding()
    {
        return patchPaddingSize;
    }

    /**
     * @return Axes order followed by the patch grid.
     */
    protected String getAxesOrder() {
        return axesOrder;
    }

    /**
     * @return Axes order used to enumerate flat patch indices.
     */
    protected String getReferenceAxesOrder() {
        return referenceAxesOrder;
    }

    /**
     * @return Grid size used to enumerate flat patch indices.
     */
    protected int[] getReferenceTileGrid() {
        return referenceTileGrid;
    }

    /**
     * Executes to string.
     *
     * @return the resulting string.
     */
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

    private static String normalizeAxesOrder(String axesOrder) {
        if (axesOrder == null || axesOrder.trim().isEmpty()) {
            return null;
        }
        return axesOrder.toLowerCase().replace(" ", "");
    }

}
