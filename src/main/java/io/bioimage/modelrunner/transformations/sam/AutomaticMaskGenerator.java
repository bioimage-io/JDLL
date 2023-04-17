package io.bioimage.modelrunner.transformations.sam;

import java.util.HashMap;
import java.util.List;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class AutomaticMaskGenerator {
	
	private int minMaskRegionArea = 1;
	private float boxNmsThres = 2;
	private float cropNmsThres = 2;
	private String outputMode;
	
	private static final String COCO_MODE = "coco_rle";
	private static final String BINARY_MODE = "binary_mask";
	private static final String RLES_KEY = "rles";
	private static final String SEGMENTATIONS_KEY = "segmentations";



	public < R extends RealType< R > & NativeType< R > > void generate( final Tensor< R > input ) {
		MaskData maskData = generateMasks(input);
		
		if (minMaskRegionArea > 0) {
			maskData = preprocessSmallRegions(maskData, minMaskRegionArea, Math.max(boxNmsThres, cropNmsThres));
		}
		
		if (outputMode.equals(COCO_MODE)) {
			// TODO
			maskData.set(SEGMENTATIONS_KEY, null)
		} else if (outputMode.equals(BINARY_MODE)) {
			maskData.set(SEGMENTATIONS_KEY, ;
		} else {
			maskData.set(SEGMENTATIONS_KEY, maskData.get(RLES_KEY));
		}
	}

	private < R extends RealType< R > & NativeType< R > > MaskData generateMasks( final Tensor< R > input ) {
		long[] dims = input.getData().dimensionsAsLongArray();
		int hInd = input.getAxesOrderString().toLowerCase().indexOf("y");
		int wInd = input.getAxesOrderString().toLowerCase().indexOf("x");
		int[] origSize = new int[] {(int) dims[hInd], (int) dims[wInd]};
		Object[] cropBoxesLayerIdxs = generateCropBoxes(origSize, this.cropNLayers, this.cropOverlapRatio);
		List<int[]> cropBoxes = (List<int[]>) cropBoxesLayerIdxs[0];
		List<Integer> layerIdxs = (List<Integer>) cropBoxesLayerIdxs[1];
		
		for (int i = 0; i < cropBoxes.size(); i ++) {
			processCrop(input, cropBoxes.get(i), layerIdxs.get(i), origSize);
		}
		return null;
	}
	
	private MaskData preprocessSmallRegions(MaskData maskData, int minArea, float nmsThres) {
		return null;
	}
	
	
	public static void rleToMask(HashMap<String, Object> rle) {
		
	}
}
