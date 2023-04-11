package io.bioimage.modelrunner.transformations;

import java.util.ArrayList;
import java.util.List;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class SamTransformation {

	private int cropNLayers = 0;
	private double cropOverlapRatio = 512 / 1500;
	
	public < R extends RealType< R > & NativeType< R > > void apply( final Tensor< R > input )
	{
		generate(input);
	}

	private < R extends RealType< R > & NativeType< R > > void generate( final Tensor< R > input ) {
		maskData = generateMasks(input);
	}

	private < R extends RealType< R > & NativeType< R > > void generateMasks( final Tensor< R > input ) {
		long[] dims = input.getData().dimensionsAsLongArray();
		int hInd = input.getAxesOrderString().toLowerCase().indexOf("y");
		int wInd = input.getAxesOrderString().toLowerCase().indexOf("x");
		int[] origSize = new int[] {hInd, wInd};
		Object[] cropBoxesLayerIdxs = generateCropBoxes(origSize, this.cropNLayers, this.cropOverlapRatio);
		List<int[]> cropBoxes = (List<int[]>) cropBoxesLayerIdxs[0];
		List<Integer> layerIdxs = (List<Integer>) cropBoxesLayerIdxs[1];
		
		for (int i = 0; i < cropBoxes.size(); i ++) {
			processCrop(input, cropBoxes.get(i), layerIdxs.get(i), origSize);
		}
	}
	
	private < R extends RealType< R > & NativeType< R > > void processCrop(final Tensor< R > image, 
			int[] cropBox, int cropLayer, int[] origSize) {
		int x0 = cropBox[0]; int y0 = cropBox[1]; int x1 = cropBox[2]; int y1 = cropBox[3];
	}
	
	private static Object[] generateCropBoxes(int[] imSize, int nLayers, double overlapRatio) {
		List<int[]> cropBoxes = new ArrayList<int[]>();
		List<Integer> layerIdxs = new ArrayList<Integer>();
		int imH = imSize[0];
		int imW = imSize[1];
		int shortSide = Math.min(imH, imW);
		cropBoxes.add(new int[] {0, 0, imW, imH});
		layerIdxs.add(0);
		
		for (int iLayer = 0; iLayer < nLayers; iLayer ++) {
			double nCropsPerSide = Math.pow(2, iLayer + 1);
			int overlap = (int) (overlapRatio *shortSide * (2 / nCropsPerSide));
			int cropW = 
					(int) (Math.ceil((overlap * (nCropsPerSide - 1) + imW) / nCropsPerSide));
			int cropH = 
					(int) (Math.ceil((overlap * (nCropsPerSide - 1) + imH) / nCropsPerSide));
			int[] cropBoxX0 = new int[(int) nCropsPerSide];
			int[] cropBoxY0 = new int[(int) nCropsPerSide];
			for (int i = 0; i < nCropsPerSide; i ++) {
				cropBoxX0[i] = (cropW - overlap) * i;
				cropBoxY0[i] = (cropH - overlap) * i;
				for (int i0 = 0; i0 < cropBoxX0.length; i ++) {
					for (int i1 = 0; i1 < cropBoxY0.length; i ++) {
						int[] box = {cropBoxX0[i0], cropBoxY0[i1],
								Math.min(cropBoxX0[i0] + cropW, imW),
								Math.min(cropBoxY0[i1] + cropH, imH)};
						cropBoxes.add(box);
						layerIdxs.add(iLayer + 1);
					}
				}
			}
		}
		return new Object[] {cropBoxes, layerIdxs};
	}
}
