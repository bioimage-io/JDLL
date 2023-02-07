package io.bioimage.modelrunner.transformations;

import io.bioimage.modelrunner.tensor.Tensor;

import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class SigmoidTransformation extends AbstractTensorPixelTransformation
{
	private static String name = "sigmoid";

	public SigmoidTransformation()
	{
		super(name);
	}

	public < R extends RealType< R > & NativeType< R > > Tensor< FloatType > apply( final Tensor< R > input )
	{
		super.setFloatUnitaryOperator( v -> ( float ) ( 1. / ( 1. + Math.exp( -v ) ) ) );
		return super.apply(input);
	}

	public void applyInPlace( final Tensor< FloatType > input )
	{
		super.setFloatUnitaryOperator( v -> ( float ) ( 1. / ( 1. + Math.exp( -v ) ) ) );
		super.apply(input);
	}
}
