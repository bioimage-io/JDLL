package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * Abstract classes for tensor transformations where a new pixel value can be
 * calculated solely from the corresponding pixel value in the input. This
 * mapping is specified by a
 *
 * @author Jean-Yves Tinevez
 *
 */
public class AbstractTensorPixelTransformation extends AbstractTensorTransformation
{

	private FloatUnaryOperator fun;

	protected AbstractTensorPixelTransformation( final String name)
	{
		super( name );
	}
	
	protected void setFloatUnitaryOperator(final FloatUnaryOperator fun) {
		this.fun = fun;
	}

	@Override
	public < R extends RealType< R > & NativeType< R > > Tensor< FloatType > apply( final Tensor< R > input )
	{
		final Tensor< FloatType > output = makeOutput( input );
		LoopBuilder
				.setImages( input.getData(), output.getData() )
				.multiThreaded()
				.forEachPixel( ( i, o ) -> o.set( fun.applyAsFloat( i.getRealFloat() ) ) );
		return output;
	}

	@Override
	public void applyInPlace( final Tensor< FloatType > input )
	{
		LoopBuilder
				.setImages( input.getData() )
				.multiThreaded()
				.forEachPixel( i -> i.set( fun.applyAsFloat( i.get() ) ) );
	}

	@FunctionalInterface
	public interface FloatUnaryOperator
	{
		float applyAsFloat( float in );
	}
}
