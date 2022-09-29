package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
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
		}
	}

	@Override
	public < R extends RealType< R > & NativeType< R > > Tensor< FloatType > apply( final Tensor< R > input )
	{
		final float[] meanStd = meanStd( input );
		final float mean = meanStd[ 0 ];
		final float std = meanStd[ 1 ];

		final Tensor< FloatType > output = makeOutput( input );
		LoopBuilder.setImages( input.getData(), output.getData() )
				.multiThreaded()
				.forEachPixel( ( i, o ) -> o.setReal( ( i.getRealDouble() - mean ) / std ) );
		return output;
	}

	@Override
	public void applyInPlace( final Tensor< FloatType > input )
	{
		final float[] meanStd = meanStd( input );
		final float mean = meanStd[ 0 ];
		final float std = meanStd[ 1 ];

		LoopBuilder.setImages( input.getData() )
				.multiThreaded()
				.forEachPixel( i -> i.set( ( i.get() - mean ) / std ) );
	}

	public static final < R extends RealType< R > & NativeType< R > > float[] meanStd( final Tensor< R > input )
	{
		// Mean.
		double sum = 0.;
		long n = 0;
		for ( final R p : Views.iterable( input.getData() ) )
		{
			sum += p.getRealDouble();
			n++;
		}
		if ( n < 1 )
			throw new IllegalArgumentException( "Tensor must contain at least 2 pixels, got " + n );

		final double mean = sum / n;

		// Variance.
		double sumdx2 = 0.;
		for ( final R p : Views.iterable( input.getData() ) )
		{
			final double dx = p.getRealDouble() - mean;
			sumdx2 += dx * dx;
		}
		final double variance = sumdx2 / ( n - 1 );
		final double std = Math.sqrt( variance );

		return new float[] { ( float ) mean, ( float ) std };
	}
}
