package io.bioimage.modelrunner.tiling;

import java.util.List;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;

public class TileCalculator {
	
	private final List<TileInfo> tileInfoList;
	
	private final ModelDescriptor descriptor;
	
	private TileCalculator(ModelDescriptor descriptor, List<TileInfo> tileInfoList) {
		this.descriptor = descriptor;
		this.tileInfoList = tileInfoList;
	}

}
