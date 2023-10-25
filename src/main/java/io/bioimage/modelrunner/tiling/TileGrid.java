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
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import io.bioimage.modelrunner.utils.IndexingUtils;

/**
 * Calculate all the coordinates for the tiles for a specific tensor in the image that wants to
 * be applied
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class TileGrid
{
	/**
	 * Size of the input patch. Following the tensor axes order
	 */
    private long[] tileSize;
    /**
     * Size of roi of each tile, following the tensor axes order
     */
    private int[] roiSize;
    
    /**
     * Name of the tensor to whom these specs correspond
     */
    private String tensorName;
    
    private List<long[]> tilePostionsInImage = new ArrayList<long[]>();
    
    private List<long[]> roiPositionsInTile = new ArrayList<long[]>();
    
    private List<long[]> roiPositionsInImage = new ArrayList<long[]>();


    private TileGrid()
    {
    }
    
    /**
     */
    public static TileGrid create(PatchSpec tileSpecs)
    {
        TileGrid ps = new TileGrid();
        ps.tensorName = tileSpecs.getTensorName();
        long[] imageDims = tileSpecs.getTensorDims();
        int[] gridSize = tileSpecs.getPatchGridSize();
        ps.tileSize = tileSpecs.getPatchInputSize();
        int tileCount = Arrays.stream(gridSize).reduce(1, (a, b) -> a * b);

        for (int j = 0; j < tileCount; j ++) {
        	int[] patchIndex = IndexingUtils.flatIntoMultidimensionalIndex(j, gridSize);
        	long[] patchSize = tileSpecs.getPatchInputSize();
        	int[][] padSize = tileSpecs.getPatchPaddingSize();
        	int[] roiSize = IntStream.range(0, patchIndex.length)
                    .map(i -> (int) patchSize[i] - padSize[0][i] - padSize[1][i]).toArray();
			ps.roiSize = roiSize;
			ps.roiPositionsInTile.add(IntStream.range(0, padSize[0].length).mapToLong(i -> (long) padSize[0][i]).toArray());
        	long[] roiStart = LongStream.range(0, patchIndex.length)
                    .map(i -> Math.min(roiSize[(int) i] * patchIndex[(int) i], imageDims[(int) i] - roiSize[(int) i])).toArray();
			ps.roiPositionsInImage.add(roiStart);
        	long[] patchStart = LongStream.range(0, patchIndex.length)
                    .map(i -> Math.min(roiSize[(int) i] * patchIndex[(int) i] - padSize[0][(int) i], imageDims[(int) i] - roiSize[(int) i]))
                    .toArray();
        	ps.tilePostionsInImage.add(patchStart);
        }
        return ps;
    }
    
    public String getTensorName() {
    	return tensorName;
    }
    
    public long[] getTileSize() {
    	return this.tileSize;
    }
    
    public int[] getRoiSize() {
    	return this.roiSize;
    }
    
    public List<long[]> getTilePostionsInImage() {
    	return this.tilePostionsInImage;
    }
    
    public List<long[]> getRoiPositionsInTile() {
    	return this.roiPositionsInTile;
    }
    
    public List<long[]> getRoiPostionsInImage() {
    	return this.roiPositionsInImage;
    }

    @Override
    public String toString()
    {
    	return "";
    	/*
    	String[] paddingStrArr = new String[patchPaddingSize[0].length];
    	for (int i = 0; i < paddingStrArr.length; i ++)
    		paddingStrArr[i] = patchPaddingSize[0][i] + "," + patchPaddingSize[1][i];
        StringBuilder builder = new StringBuilder();
        builder.append("PatchSpec of '" + tensorName + "'"
        		+ "[patchInputSize=").append(Arrays.toString(this.tileSize))
                .append(", patchGridSize=").append(Arrays.toString(patchGridSize))
                .append(", patchPaddingSize=").append(Arrays.toString(paddingStrArr))
                .append("]");
        return builder.toString();
        */
    }

}
