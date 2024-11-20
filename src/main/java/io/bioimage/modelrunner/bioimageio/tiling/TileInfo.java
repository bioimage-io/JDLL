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

public class TileInfo {
	
	
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
	
	

}
