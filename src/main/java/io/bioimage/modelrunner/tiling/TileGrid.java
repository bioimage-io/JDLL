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
	 * Size of the tile for a tensor. Following the tensor axes order
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
    
    /**
     * The position of the closest corner to the center for each of the tiles  with 
     * respect to the original image of the tensor.
     * The positions might be negative as the image that is going to be processed might have padding on the edges
     */
    private List<long[]> tilePostionsInImage = new ArrayList<long[]>();
    
    /**
     * The positions of the roi in each of the tiles that are going to be used
     */
    private List<long[]> roiPositionsInTile = new ArrayList<long[]>();
    
    /**
     * The positions of the roi of each of the tiles in the original image. Where the ROIs are
     * inserted on the original image.
     */
    private List<long[]> roiPositionsInImage = new ArrayList<long[]>();


    private TileGrid()
    {
    }
    
    /**
     * Calculate the positions of each of the tiles that are going to be used to process the whole image.
     * 
     * @param tileSpecs
     * 	instance of {@link PatchSpec} that contains the info about the size of the original image, and the 
     * 	size of the tiles that are going to be used
     * @return an instance of {@link TileGrid} that contains the information about where the tiles need to be 
     * 	positioned with respect to the original image
     */
    public static TileGrid create(PatchSpec tileSpecs)
    {
        TileGrid ps = new TileGrid();
        ps.tensorName = tileSpecs.getTensorName();
        long[] imageDims = tileSpecs.getNonTiledTensorDims();
        int[] gridSize = tileSpecs.getTileGrid();
        ps.tileSize = tileSpecs.getTileSize();
        int tileCount = Arrays.stream(gridSize).reduce(1, (a, b) -> a * b);

        for (int j = 0; j < tileCount; j ++) {
        	int[] patchIndex = IndexingUtils.flatIntoMultidimensionalIndex(j, gridSize);
        	long[] patchSize = tileSpecs.getTileSize();
        	int[][] padSize = tileSpecs.getPadding();
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
    
    /**
     * 
     * @return name of the tensor that is going to be processed
     */
    public String getTensorName() {
    	return tensorName;
    }
    
    /**
     * 
     * @return size of the tile that is going to be used to process the image
     */
    public long[] getTileSize() {
    	return this.tileSize;
    }
    
    /**
     * 
     * @return size of the roi of each of the tiles that is going to be used to process the image
     */
    public int[] getRoiSize() {
    	return this.roiSize;
    }
    
    /**
     * 
     * @return the position of the closest corner to the center for each of the tiles  with 
     * respect to the original image of the tensor.
     * The positions might be negative as the image that is going to be processed might have padding on the edges
     */
    public List<long[]> getTilePostionsInImage() {
    	return this.tilePostionsInImage;
    }
    
    /**
     * 
     * @return the positions of the roi in each of the tiles that are going to be used
     */
    public List<long[]> getRoiPositionsInTile() {
    	return this.roiPositionsInTile;
    }
    
    /**
     * 
     * @return the positions of the roi of each of the tiles in the original image. Where the ROIs are
     * inserted on the original image.
     */
    public List<long[]> getRoiPostionsInImage() {
    	return this.roiPositionsInImage;
    }

}
