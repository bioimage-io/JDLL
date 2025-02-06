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
import java.util.List;

public class TileInfo {

	
	private long[] halo;
	
	private String haloAxesOrder;
	
	private final String name;
	
	private final long[] imDims;
	
	private final long[] proposedTileDims;
	
	private final String imAxesOrder;
	
	private final String tileAxesOrder;
	
	private TileInfo(String tensorName, long[] imDims, String imAxesOrder, long[] proposedTileDims, String tileAxesOrder) {
		this.name = tensorName;
		this.imAxesOrder = imAxesOrder;
		this.imDims = imDims;
		this.proposedTileDims = proposedTileDims;
		this.tileAxesOrder = tileAxesOrder;
		this.halo = new long[imAxesOrder.length()];
	}
	
	public static TileInfo build(String tensorName, long[] imDims, String imAxesOrder, long[] proposedTileDims, String tileAxesOrder) {
		return new TileInfo(tensorName, imDims, imAxesOrder, proposedTileDims, tileAxesOrder);
	}

	/**
	 * @return the name
	 */
	public String getName() {
		return name;
	}

	/**
	 * @return the imDims
	 */
	public long[] getImageDims() {
		return imDims;
	}

	/**
	 * @return the proposedTileDims
	 */
	public long[] getTileDims() {
		return proposedTileDims;
	}

	/**
	 * @return the imAxesOrder
	 */
	public String getImageAxesOrder() {
		return imAxesOrder;
	}

	/**
	 * @return the tileAxesOrder
	 */
	public String getTileAxesOrder() {
		return tileAxesOrder;
	}
	
	/**
	 * 
	 * @return the halo wanted
	 */
	public long[] getHalo() {
		return this.halo;
	}
	
	/**
	 * 
	 * @return the halo axesOrder
	 */
	public String getHaloAxesOrder() {
		return this.haloAxesOrder;
	}
	
	
	/**
	 * How many edge pixels per dimension you need to discard. For example if the model has artifacts
	 * on up to the third pixel on each side of the plane XY, these pixels will be discarded and replaced
	 * by others using tiling.
	 * The halo should only be specified for the outputs.
	 * 
	 * If the model requires tensors of dimensions "bcyx", the halo argument should be something like: {0, 0, 3, 3}.
	 * 
	 * In this case, a total of 6 pixels (3 per side) is removed
	 * 
	 * The halo should never be bigger than half of the 
	 * 
	 * Both argument should have the same length.
	 * 
	 * @param halo
	 * 	number of pixels per dimension per edge to remove
	 * @param haloAxesOrder
	 * 	axes order of the specified halo
	 */
	public void setHalo(long[] halo, String haloAxesOrder) {
		if (halo.length != haloAxesOrder.length())
			throw new IllegalArgumentException("The halo array and axes order should have the same length: "
					+ Arrays.asList(halo) + " (" + halo.length + ") vs " + haloAxesOrder + "(" + haloAxesOrder.length() + ").");
		for (String ax : haloAxesOrder.split("")) {
			if (!this.tileAxesOrder.toLowerCase().contains(ax.toLowerCase()))
				throw new IllegalArgumentException("Dimension '" + ax + "' is not present in the tile axes order (" + this.tileAxesOrder + ")");
		}
		this.halo = halo;
		this.haloAxesOrder = haloAxesOrder;
	}
	
	/**
	 * If the model has several outputs with different halos each. For each tiles calculates the halo
	 * that allows every output to be processed with the same number of tiles
	 * @param tiles
	 * 	the tiles of interest. The halo value will be modified inplace after running the method
	 */
	public static void adaptHalos(List<TileInfo> tiles) {
		String axesOrder = "";
		List<Long> halo = new ArrayList<Long>();
		List<Long> dims = new ArrayList<Long>();
		for (TileInfo tile : tiles) {
			for (String ax : tile.haloAxesOrder.split("")) {
				if (axesOrder.indexOf(ax) == -1) {
					axesOrder += ax;
					halo.add(-1L);
					dims.add(-1L);
				}
			}
			int c = 0;
			for (String ax : tile.haloAxesOrder.split("")) {
				int ind = axesOrder.indexOf(ax);
				int ind2 = tile.tileAxesOrder.indexOf(ax);
				if (halo.get(ind) == -1) {
					halo.set(ind, tile.getHalo()[c]);
					dims.set(ind, tile.getTileDims()[ind2]);
				} else {
					Long hh = halo.get(ind);
					Long dd = dims.get(ind);
					double ratio = dd / (double) tile.getTileDims()[ind2];
					double nh = Math.floor(tile.getHalo()[c] * ratio);
					if (nh < hh)
						tile.halo[c] = (long) Math.floor(hh / ratio);
					else if (nh > hh)
						tile.halo[c] = (long) nh;
				}
			}
		}
	}

}
