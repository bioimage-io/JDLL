package io.bioimage.modelrunner.transformations;

import java.util.ArrayList;
import java.util.List;

import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.model.Model;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccessible;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class SamTransformation {

	private static float eps = (float) Math.pow(10, -6);
	
	private int cropNLayers = 0;
	private double cropOverlapRatio = 512 / 1500;
	private long imageSize = 1024;
	private boolean isImageSet = false;
    private Object features = null;
    private int origH = 0;
    private int origW = 0;
    private int inputH = 0;
    private int inputW = 0;
    private long[] inputSize;
    private long[] originalSize;
	
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
		String axes = image.getAxesOrderString().toLowerCase();
		int hInd = axes.indexOf("y");
		int wInd = axes.indexOf("x");
		long[] start = new long[axes.length()];
		long[] end = new long[axes.length()];
		start[hInd] = y0; start[wInd] = x0;
		end[hInd] = y1; end[wInd] = x1;
		IntervalView<R> croppedIm = Views.interval(image.getData(), start, end);
		int[] croppedImSize = new int[] {y1 - y0, x1 - x0};
		long[] tensorShape = croppedIm.dimensionsAsLongArray();
    	final ArrayImgFactory< R > factory = new ArrayImgFactory<>( image.getData().getAt(0) );
        final Img< R > croppedIm2 = (Img<R>) factory.create(tensorShape);

		LoopBuilder.setImages( image.getData(), croppedIm2 )
				.multiThreaded()
				.forEachPixel( (i, j) -> j.set( i ));
		resize(croppedIm2, imageSize, axes);
	}
	
	
	private < R extends RealType< R > & NativeType< R > > void setTorchImage(Img<R> image) throws LoadEngineException, Exception {
		resetImage();
		inputSize = new long[] {image.dimensionsAsLongArray()[2], image.dimensionsAsLongArray()[2]}; 
		preprocess((Img<FloatType>) image);
		
		// TODO get encoder and run model
		EngineInfo engineInfo = EngineInfo.defineDLEngine("pytorch", "1.13.1", true, true);
		Model model = Model.createDeepLearningModel("path/to/cache", "path/to/cache/model.pt", engineInfo);
		model.loadModel();
		// TODO Create input and output tensor lists
		// TODO Create input and output tensor lists
		// TODO Create input and output tensor lists
		// TODO Create input and output tensor lists
		// TODO Create input and output tensor lists
		model.runModel(null, null);
		isImageSet = true;
		
	}
	
	public static void preprocess(Img<FloatType> image) {
		final float[] meanStd = ZeroMeanUnitVarianceTransformation.meanStd( image );
		final float mean = meanStd[ 0 ];
		final float std = meanStd[ 1 ];
		LoopBuilder.setImages( image ).multiThreaded()
			.forEachPixel( i -> i.set( ( i.get() - mean ) / ( std + eps ) ) );
		// TODO add padding
		// TODO add padding
		// TODO add padding
		// TODO add padding
		// TODO add padding
	}
	
	private < R extends RealType< R > & NativeType< R > > void resize(Img<R> image, long targetLength, String axes) {
		int hInd = axes.indexOf("y"); int wInd = axes.indexOf("x");
		int[] targetSize = getPreprocessShape(image.dimensionsAsLongArray()[hInd], 
				image.dimensionsAsLongArray()[wInd], targetLength);
		NLinearInterpolatorFactory<R> interpFactory = new NLinearInterpolatorFactory<R>();
		
		// TODO TODO TODO review this
		// TODO TODO TODO review this
		// TODO TODO TODO review this
		// TODO TODO TODO review this
		// TODO TODO TODO review this interpolation
		
		RealRandomAccessible<R> interpolated = Views.interpolate(image, interpFactory);
	}
	
	private static int[] getPreprocessShape(long oldH, long oldW, long longSideLength) {
		double scale = ((double) longSideLength) * 1 / Math.max(oldW, oldH);
		double newh = oldH * scale; double neww = oldW * scale;
		int neww2 = (int) Math.floor(neww + 0.5);
		int newh2 = (int) Math.floor(newh + 0.5);
		return new int[] {newh2, neww2};
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
	
	private void resetImage() {
	    origH = 0;
	    origW = 0;
	    inputH = 0;
	    inputW = 0;
	}
}
