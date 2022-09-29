package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class ZeroMeanUnitVarianceTensorTransformation extends AbstractTensorTransformation
{
	
	private static String name = "zero_mean_unit_variace";
	private Double meanVal;
	private Double stdVal;
	private double[] meanArr;
	private double[] stdArr;
	private Mode mode = Mode.PER_SAMPLE;
	private String axes;

	public ZeroMeanUnitVarianceTensorTransformation()
	{
		super( name );
	}
	
	public void getMean(double mean) {
		this.meanVal = mean;
	}
	
	public void getMean(double[] meanArr) {
		this.meanArr = meanArr;
	}
	
	public void getStd(double std) {
		this.stdVal = std;
	}
	
	public void getStd(double[] std) {
		this.stdArr = std;
	}
	
	public void getAxes(String axes) {
		this.axes = axes;
	}
	
	public void getMode(String mode) {
		this.mode = Mode.valueOf(mode);
	}
	
	public void getMode(Mode mode) {
		this.mode = mode;
	}
	
	private void calculateMeanStd() {
		
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
		String selectedAxes = "";
		for (String ax : input.getAxesOrderString().split("")) {
			if (axes != null && !axes.toLowerCase().contains(ax.toLowerCase())
					&& !ax.toLowerCase().equals("b"))
				selectedAxes += ax;
		}
		if (mode == Mode.FIXED &&  (axes == null || selectedAxes.equals("") 
				|| input.getAxesOrderString().replace("b", "").length() - selectedAxes.length() == 1)) {
			if (meanVal == null)
				throw new IllegalArgumentException("The 'axes' parameter is not"
						+ " compatible with the parameters 'mean' and 'std' if 'mode' is 'fixed'."
						+ "The parameters 'mean' and 'std' cannot be arrays with the introduced 'axes'.");
			fixedModeGlobalMeanStd(output);
			return output;
		} else if (mode != Mode.FIXED && (axes == null || selectedAxes.equals("") 
				|| input.getAxesOrderString().replace("b", "").length() - selectedAxes.length() == 1)) {
			if (meanVal == null)
				throw new IllegalArgumentException("The 'axes' parameter is not"
						+ " compatible with the parameters 'mean' and 'std' if 'mode' is 'fixed'."
						+ "The parameters 'mean' and 'std' cannot be arrays with the introduced 'axes'.");
			notFixedModeGlobalMeanStd(output);
			return output;
		} else if (mode != Mode.FIXED 
				&& input.getAxesOrderString().replace("b", "").length() - selectedAxes.length() == 2) {
			notFixedAxesMeanStd(output, selectedAxes);
			return output;
		} else if (mode == Mode.FIXED 
				&& input.getAxesOrderString().replace("b", "").length() - selectedAxes.length() == 2) {
			fixedAxesMeanStd(output, selectedAxes);
			return output;
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
			IntervalView<FloatType> plane = Views.interval( output.getData(), start, dims );
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
			IntervalView<FloatType> plane = Views.interval( output.getData(), start, dims );
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

	@Override
	public void applyInPlace( final Tensor< FloatType > input )
	{
		final float[] meanStd = meanStd( input.getData() );
		final float mean = meanStd[ 0 ];
		final float std = meanStd[ 1 ];

		LoopBuilder.setImages( input.getData() )
				.multiThreaded()
				.forEachPixel( i -> i.set( ( i.get() - mean ) / std ) );
	}

	public static float[] meanStd( final RandomAccessibleInterval< FloatType > rai )
	{
		// Mean.
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
		getAllCombinations(new long[] {6,1,4});
	}
}
