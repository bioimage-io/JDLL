package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

public class ZeroMeanUnitVarianceTensorTransformation extends AbstractTensorTransformation
{

	public ZeroMeanUnitVarianceTensorTransformation()
	{
		super( "sigmoid" );
	}

	@Override
	public < R extends RealType< R > & NativeType< R >, Q extends RealType< Q > & NativeType< Q > > void apply( final Tensor< R > input, final Tensor< Q > output )
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

		LoopBuilder.setImages( input.getData(), output.getData() )
				.multiThreaded()
				.forEachPixel( ( i, o ) -> o.setReal( ( i.getRealDouble() - mean ) / std ) );
	}
}
