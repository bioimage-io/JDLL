package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class ClipTensorTransformation extends AbstractTensorTransformation
{

	private final double min;

	private final double max;

	public ClipTensorTransformation( final double min, final double max )
	{
		super( "clip( " + Math.min( min, max ) + ", " + Math.max( min, max ) + " )" );
		this.min = Math.min( min, max );
		this.max = Math.max( min, max );
	}

	@Override
	public < R extends RealType< R > & NativeType< R >, Q extends RealType< Q > & NativeType< Q > > void apply( final Tensor< R > input, final Tensor< Q > output )
	{
		LoopBuilder.setImages( input.getData(), output.getData() )
				.multiThreaded()
				.forEachPixel( ( i, o ) -> o.setReal(
						i.getRealDouble() > max
								? max
								: i.getRealDouble() < min
										? min
										: i.getRealDouble() ) );
	}
}
