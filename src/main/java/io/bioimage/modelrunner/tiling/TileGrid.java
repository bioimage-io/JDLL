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
import java.util.stream.IntStream;

import org.bioimageanalysis.icy.deepicy.tools.ImgLib2Utils;

import io.bioimage.modelrunner.utils.IndexingUtils;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.util.Util;

/**
 * Calculate all the coordinates for the tiles for a specific tensor in the image that wants to
 * be applied
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class TileGrid
{
	/**
	 * Size of the input patch. Following "xyczb" axes order
	 */
    private int[] patchInputSize;
    /**
     * Size of the number of patches per axis. Following "xyczb" axes order
     */
    private int[] patchGridSize;
    
    private int[][] corner;
    /**
     * Name of the tensor to whom these specs correspond
     */
    private String tensorName;

    /**
     */
    public static TileGrid create(PatchSpec tileSpecs, long[] imSize, int[] gridSize)
    {
        TileGrid ps = new TileGrid();
        int tileCount = Arrays.stream(gridSize).reduce(1, (a, b) -> a * b);

        for (int j = 0; j < tileCount; j ++) {
        	int[] patchIndex = IndexingUtils.flatIntoMultidimensionalIndex(j, gridSize);
        	int[] patchSize = tileSpecs.getPatchInputSize();
        	int[][] padSize = tileSpecs.getPatchPaddingSize();
        	int[][] padSizeSeparated = new int[padSize.length][padSize[0].length];
        	int[] roiSize = IntStream.range(0, patchIndex.length)
                    .map(i -> patchSize[i] - padSizeSeparated[0][i] - padSizeSeparated[1][i]).toArray();
        	int[] patchStart = IntStream.range(0, patchIndex.length)
                    .map(i -> roiSize[i] * patchIndex[i] - padSizeSeparated[0][i]).toArray();
        }
        
        
        
        
        String patchAxesOrder = "xyczb";
        int[] patchSize = tileSpecs.getPatchInputSize();
        int[] patchSizeTensorAxes = PatchGridCalculator.arrayToWantedAxesOrderAddOnes(patchSize, patchAxesOrder, tensor.getAxesOrder());
        int[] patchIndexTensorAxes = PatchGridCalculator.arrayToWantedAxesOrderAddOnes(patchIndex, patchAxesOrder, tensor.getAxesOrder());
        int[][] padSize = tileSpecs.getPatchPaddingSize();
        int[][] padSizeAxesTensorAxes = new int[padSize.length][padSize[0].length];
        padSizeAxesTensorAxes[0] = PatchGridCalculator.arrayToWantedAxesOrderAddOnes(padSize[0], patchAxesOrder, tensor.getAxesOrder());
        padSizeAxesTensorAxes[1] = PatchGridCalculator.arrayToWantedAxesOrderAddOnes(padSize[1], patchAxesOrder, tensor.getAxesOrder());

        int[] roiSizeAxesOrder = IntStream.range(0, patchIndexTensorAxes.length)
                .map(i -> patchSizeTensorAxes[i] - padSizeAxesTensorAxes[0][i] - padSizeAxesTensorAxes[1][i]).toArray();
        int[] patchStartAxesOrder = IntStream.range(0, patchIndexTensorAxes.length)
                .map(i -> roiSizeAxesOrder[i] * patchIndexTensorAxes[i] - padSizeAxesTensorAxes[0][i]).toArray();

        Img< T > patchSequence = new ArrayImgFactory((NativeType) Util.getTypeFromInterval(inputNDArray))
                .create( patchSizeTensorAxes);
        ImgLib2Utils.copyRaiData(inputNDArray, patchSequence, patchStartAxesOrder, padSizeAxesTensorAxes[1]);
        ImgLib2Utils.addMirrorToPatchRai(inputNDArray, patchSequence, patchStartAxesOrder, padSizeAxesTensorAxes);
        
        return ps;
    }

    private TileGrid()
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
    public static int[] getGridSize(List<TileGrid> patches) {
    	// The minimum possible grid is just one patch in every direction. This is the
    	// grid if no tiling is allowed
    	int[] grid = new int[]{1, 1, 1, 1, 1};
    	// If there is any different grid, that will be the absolute one
    	for (TileGrid pp : patches) {
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
    public static int[] getGridSize(Map<String, TileGrid> patches) {
    	// The minimum possible grid is just one patch in every direction. This is the
    	// grid if no tiling is allowed
    	int[] grid = new int[]{1, 1, 1, 1, 1};
    	// If there is any different grid, that will be the absolute one
    	for (TileGrid pp : patches.values()) {
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
    public static TileGrid getPatchSpecFromListByName(List<TileGrid> specs, String name) {
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
     * @return Input patch size. The patch taken from the input sequence including the halo.
     */
    public int[] getPatchInputSize()
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
