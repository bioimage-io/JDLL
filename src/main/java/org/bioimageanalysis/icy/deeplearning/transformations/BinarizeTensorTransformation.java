package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class BinarizeTensorTransformation extends AbstractTensorTransformation
{

	private final double threshold;

	public BinarizeTensorTransformation( final double threshold )
	{
		super( "binarize( " + threshold + " )" );
		this.threshold = threshold;
	}

	@Override
	public < R extends RealType< R > & NativeType< R >, Q extends RealType< Q > & NativeType< Q > > void apply( final Tensor< R > input, final Tensor< Q > output )
	{
		LoopBuilder.setImages( input.getData(), output.getData() )
				.multiThreaded()
				.forEachPixel( ( i, o ) -> o.setReal( i.getRealDouble() >= threshold ? 1. : 0. ) );
	}
}
