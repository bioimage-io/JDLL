package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class ClipTensorTransformation extends AbstractTensorPixelTransformation
{

	private static final class ClipFunction implements FloatUnaryOperator
	{

		private final float min;

		private final float max;

		private ClipFunction( final double min, final double max )
		{
			this.min = (float) min;
			this.max = (float) max;
		}

		@Override
		public final float applyAsFloat( final float in )
		{
			return ( in > max )
					? max
					: ( in < min )
							? min
							: in;
		}
	}
	
	private static String name = "clip";
	private Double min;
	private Double max;

	public ClipTensorTransformation()
	{
		super(name);
	}
	
	public void setMin(Double min) {
		this.min = min;
	}
	
	public void setMax(Double max) {
		this.max = max;
	}
	
	public void checkRequiredArgs() {
		if (min == null) {
			throw new IllegalArgumentException(String.format(DEFAULT_MISSING_ARG_ERR, "min"));
		} else if (max == null) {
			throw new IllegalArgumentException(String.format(DEFAULT_MISSING_ARG_ERR, "max"));
		}
	}

	public < R extends RealType< R > & NativeType< R > > Tensor< FloatType > apply( final Tensor< R > input )
	{
		checkRequiredArgs();
		super.setFloatUnitaryOperator(new ClipFunction( min, max ) );
		return super.apply(input);
	}

	public void applyInPlace( final Tensor< FloatType > input )
	{
		checkRequiredArgs();
		super.setFloatUnitaryOperator(new ClipFunction( min, max ) );
		super.apply(input);
	}
}
