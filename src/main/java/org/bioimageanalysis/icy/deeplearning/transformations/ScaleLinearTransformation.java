package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class ScaleLinearTransformation extends AbstractTensorPixelTransformation
{

	private static final class ScaleFunction implements FloatUnaryOperator
	{

		private final float gain;

		private final float offset;

		public ScaleFunction( final double gain, final double offset )
		{
			this.gain = ( float ) gain;
			this.offset = ( float ) offset;
		}

		@Override
		public final float applyAsFloat( final float in )
		{
			return gain * in + offset;
		}
	}

	private static final String name = "scale_linear";
	private final Double gain;
	private final Double offset;
	private final String axes;

	public ScaleLinearTransformation()
	{
		super( name );
	}
	
	public void setGain(double gain) {
		this.gain = gain;
	}
	
	public void setOffset(double offset) {
		this.offset = offset;
	}
	
	public void checkRequiredArgs() {
		if (offset == null) {
			throw new IllegalArgumentException(String.format(DEFAULT_MISSING_ARG_ERR, "offset"));
		} else if (gain == null) {
			throw new IllegalArgumentException(String.format(DEFAULT_MISSING_ARG_ERR, "gain"));
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
