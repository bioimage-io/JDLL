package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class ZeroMeanUnitVarianceTransformation extends AbstractTensorTransformation
{
	
	private static String name = "zero_mean_unit_variace";
	private Double meanVal;
	private Double stdVal;
	private double[] meanArr;
	private double[] stdArr;
	private Mode mode = Mode.PER_SAMPLE;
	private String axes;

	private static String FIXED_MODE_ERR = "If the mode is 'fixed', the parameters 'mean' and"
			+ " 'std need to be specified";
	private static String NOT_FIXED_MODE_ERR = "Only the mode 'fixed' requires providing the "
			+ "'std' and 'mean parameters.";

	public ZeroMeanUnitVarianceTransformation()
	{
		super( name );
	}
	
	public void setMean(double mean) {
		this.meanVal = mean;
	}
	
	public void setMean(double[] meanArr) {
		this.meanArr = meanArr;
	}
	
	public void setMean(double[][] meanArr) {
		//TODO allow scaling of more complex structures
		throw new IllegalArgumentException("At the moment, only allowed calculations on planes.");
	}
	
	public void setStd(double std) {
		this.stdVal = std;
	}
	
	public void setStd(double[] std) {
		this.stdArr = std;
	}
	
	public void setStd(double[][] std) {
		//TODO allow scaling of more complex structures
		throw new IllegalArgumentException("At the moment, only allowed calculations on planes.");
	}
	
	public void setAxes(String axes) {
		this.axes = axes;
	}
	
	public void setMode(String mode) {
		this.mode = Mode.valueOf(mode.toUpperCase());
	}
	
	public void setMode(Mode mode) {
		this.mode = mode;
	}
	
	public void checkRequiredArgs() {
		if (this.mode == Mode.FIXED && this.meanArr == null && this.meanVal == null) {
			throw new IllegalArgumentException(String.format(DEFAULT_MISSING_ARG_ERR, "mean")
					+ System.lineSeparator() + "If 'mode' parameter equals 'fixed', the 'mean' "
							+ "argument should be provided too.");
		} else if (this.mode == Mode.FIXED && this.stdArr == null && this.stdVal == null) {
			throw new IllegalArgumentException(String.format(DEFAULT_MISSING_ARG_ERR, "std")
					+ System.lineSeparator() + "If 'mode' parameter equals 'fixed', the 'std' "
					+ "argument should be provided too.");
		} else if (this.mode == Mode.FIXED && ((stdVal == null && meanVal != null)
				|| (stdVal != null && meanVal == null))) {
			throw new IllegalArgumentException("Both arguments 'mean' and "
					+ "'std' need to be of the same type. Either a single value or an array.");
		} else if (this.mode == Mode.FIXED && this.meanArr != null && axes == null) {
			throw new IllegalArgumentException("If 'mean' and 'std' are provided as arrays "
					+ "and 'mode' is 'fixed', the corresponding 'axes' argument should be provided too.");
		}
	}

	@Override
	public < R extends RealType< R > & NativeType< R > > Tensor< FloatType > apply( final Tensor< R > input )
	{
		checkRequiredArgs();
		final Tensor< FloatType > output = makeOutput( input );
		applyInPlace(output);
		return output;
	}

	@Override
	public void applyInPlace( final Tensor< FloatType > input )
	{
		checkRequiredArgs();
		String selectedAxes = "";
		for (String ax : input.getAxesOrderString().split("")) {
			if (axes != null && !axes.toLowerCase().contains(ax.toLowerCase())
					&& !ax.toLowerCase().equals("b"))
				selectedAxes += ax;
		}
		if (mode == Mode.FIXED &&  (axes == null || selectedAxes.equals("") 
				|| input.getAxesOrderString().replace("b", "").length() == selectedAxes.length())) {
			if (meanVal == null && meanArr == null)
				throw new IllegalArgumentException(FIXED_MODE_ERR);
			else if (meanVal == null)
				throw new IllegalArgumentException("The parameters 'mean' and 'std' "
						+ "cannot be arrays with the introduced 'axes'.");
			fixedModeGlobalMeanStd(input);
		} else if (mode != Mode.FIXED && (axes == null || selectedAxes.equals("") 
				|| input.getAxesOrderString().replace("b", "").length() == selectedAxes.length())) {
			if (meanVal != null || meanArr != null)
				throw new IllegalArgumentException(NOT_FIXED_MODE_ERR);
			notFixedModeGlobalMeanStd(input);
		} else if (mode != Mode.FIXED 
				&& axes.length() <= 2 && axes.length() > 0) {
			if (meanVal != null || meanArr != null)
				throw new IllegalArgumentException(NOT_FIXED_MODE_ERR);
			notFixedAxesMeanStd(input, selectedAxes);
		} else if (mode == Mode.FIXED 
				&& axes.length() <= 2 && axes.length() > 0) {
			if (meanVal == null && meanArr == null)
				throw new IllegalArgumentException(FIXED_MODE_ERR);
			else if (meanVal != null)
				throw new IllegalArgumentException("The parameters 'mean' and ' std' "
						+ "have to be arrays with the introduced 'axes'.");
			fixedAxesMeanStd(input, selectedAxes);
		} else {
			//TODO allow scaling of more complex structures
			throw new IllegalArgumentException("At the moment, only allowed scaling of planes.");
		}
	}
	
	private void fixedModeGlobalMeanStd( final Tensor< FloatType > output ) {

		LoopBuilder.setImages( output.getData() )
				.multiThreaded()
				.forEachPixel( i -> i.set( ( i.get() - meanVal.floatValue() ) / stdVal.floatValue() ) );
	}
	
	private void notFixedAxesMeanStd( final Tensor< FloatType > output, String axesOfInterest) {
		long[] start = new long[output.getData().numDimensions()];
		long[] dims = output.getData().dimensionsAsLongArray();
		long[] indOfDims = new long[axesOfInterest.length()];
		long[] sizeOfDims = new long[axesOfInterest.length()];
		for (int i = 0; i < indOfDims.length; i ++) {
			indOfDims[i] = output.getAxesOrderString().indexOf(axesOfInterest.split("")[i]);
		}
		for (int i = 0; i < sizeOfDims.length; i ++) {
			sizeOfDims[i] = dims[(int) indOfDims[i]];
		}
		
		long[][] points = getAllCombinations(sizeOfDims);
		
		for (long[] pp : points) {
			for (int i = 0; i < pp.length; i ++) {
				start[(int) indOfDims[i]] = pp[i];
				dims[(int) indOfDims[i]] = pp[i] + 1;
			}
			// Define the view by defining the length per axis
			long[] end = new long[dims.length];
			for (int i = 0; i < dims.length; i ++) end[i] = dims[i] - start[i];
			IntervalView<FloatType> plane = Views.offsetInterval( output.getData(), start, end );
			final float[] meanStd = meanStd( plane );
			final float mean = meanStd[ 0 ];
			final float std = meanStd[ 1 ];
			LoopBuilder.setImages( plane )
					.multiThreaded()
					.forEachPixel( i -> i.set( ( i.get() - mean ) / std ) );
		}
	}
	
	private void fixedAxesMeanStd( final Tensor< FloatType > output, String axesOfInterest) {
		long[] start = new long[output.getData().numDimensions()];
		long[] dims = output.getData().dimensionsAsLongArray();
		long[] indOfDims = new long[axesOfInterest.length()];
		long[] sizeOfDims = new long[axesOfInterest.length()];
		for (int i = 0; i < indOfDims.length; i ++) {
			indOfDims[i] = output.getAxesOrderString().indexOf(axesOfInterest.split("")[i]);
		}
		for (int i = 0; i < sizeOfDims.length; i ++) {
			sizeOfDims[i] = dims[(int) indOfDims[i]];
		}
		
		long[][] points = getAllCombinations(sizeOfDims);
		int c = 0;
		for (long[] pp : points) {
			for (int i = 0; i < pp.length; i ++) {
				start[(int) indOfDims[i]] = pp[i];
				dims[(int) indOfDims[i]] = pp[i] + 1;
			}
			// Define the view by defining the length per axis
			long[] end = new long[dims.length];
			for (int i = 0; i < dims.length; i ++) end[i] = dims[i] - start[i];
			IntervalView<FloatType> plane = Views.offsetInterval( output.getData(), start, end );
			final float mean = (float) this.meanArr[c];
			final float std = (float) this.stdArr[c ++ ];
			LoopBuilder.setImages( plane )
					.multiThreaded()
					.forEachPixel( i -> i.set( ( i.get() - mean ) / std ) );
		}
	}
	
	private void notFixedModeGlobalMeanStd( final Tensor< FloatType > output ) {

		final float[] meanStd = meanStd( output.getData() );
		final float mean = meanStd[ 0 ];
		final float std = meanStd[ 1 ];
		LoopBuilder.setImages( output.getData() )
				.multiThreaded()
				.forEachPixel( i -> i.set( ( i.get() - mean ) / std ) );
	}

	public static float[] meanStd( final RandomAccessibleInterval< FloatType > rai )
	{
		// Mean.
		double aa = Util.average(Util.asDoubleArray(rai));
		double sum = 0.;
		long n = 0;
		for ( final FloatType p : Views.iterable( rai ) )
		{
			sum += p.getRealDouble();
			n++;
		}
		if ( n < 1 )
			throw new IllegalArgumentException( "Tensor must contain at least 2 pixels, got " + n );

		final double mean = sum / n;

		// Variance.
		double sumdx2 = 0.;
		for ( final FloatType p : Views.iterable( rai ) )
		{
			final double dx = p.getRealDouble() - mean;
			sumdx2 += dx * dx;
		}
		final double variance = sumdx2 /  n ;
		final double std = Math.sqrt( variance );

		return new float[] { ( float ) mean, ( float ) std };
	}
	
	private static long[][] getAllCombinations(long[] arr){
		long n = 1;
		for (long nn : arr) n *= nn;
		long[][] allPoints = new long[(int) n][arr.length];
		for (int i = 0; i < n; i ++) {
			for (int j = 0; j < arr.length; j ++) {
				int factor = 1;
				for (int k = 0; k < j; k ++) {
					factor *= arr[k];
				}
				int auxVal = i / factor;
				int val = auxVal % ((int) arr[j]);
				allPoints[i][j] = val;
			}
		}
		return allPoints;
	}
	
	public static void main(String[] args) {
		test1();
		test2();
		test3();
	}
	
	public static void test1() {
		float[] arr = new float[9];
		for (int i = 0; i < arr.length; i ++) {
			arr[i] = i;
		}
		ZeroMeanUnitVarianceTransformation preprocessing = new ZeroMeanUnitVarianceTransformation();
		preprocessing.setMean(4);
		preprocessing.setStd(4);
		preprocessing.setMode("fixed");
		ArrayImg<FloatType, FloatArray> rai = ArrayImgs.floats(arr, new long[] {3, 3});
		Tensor<FloatType> tt = Tensor.build("name", "xy", rai);
		preprocessing.applyInPlace(tt);
		System.out.print(true);
	}
	
	public static void test2() {
		float[] arr = new float[18];
		for (int i = 0; i < arr.length; i ++) {
			arr[i] = i;
		}
		ArrayImg<FloatType, FloatArray> rai = ArrayImgs.floats(arr, new long[] {3, 3, 2});
		ZeroMeanUnitVarianceTransformation preprocessing = new ZeroMeanUnitVarianceTransformation();
		preprocessing.setAxes("xy");
		preprocessing.setMode("per_sample");
		Tensor<FloatType> tt = Tensor.build("name", "xyc", rai);
		preprocessing.applyInPlace(tt);
		System.out.print(true);
	}
	
	public static void test3() {
		float[] arr = new float[9];
		for (int i = 0; i < arr.length; i ++) {
			arr[i] = i;
		}
		ArrayImg<FloatType, FloatArray> rai = ArrayImgs.floats(arr, new long[] {1, 1, 3, 3});
		ZeroMeanUnitVarianceTransformation preprocessing = new ZeroMeanUnitVarianceTransformation();
		preprocessing.setAxes("y");
		preprocessing.setMode("fixed");
		preprocessing.setMean(new double[] {1, 4, 7});
		preprocessing.setStd(new double[] {0.81650, 0.81650, 0.81650});
		Tensor<FloatType> tt = Tensor.build("name", "bcyx", rai);
		preprocessing.applyInPlace(tt);
		System.out.print(true);
	}
}
