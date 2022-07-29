package org.bioimageanalysis.icy.deeplearning.transformations;

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

	public static final String name = "scale_linear";

	public ScaleLinearTransformation( final double gain, final double offset )
	{
		super( name, new ScaleFunction( gain, offset ) );
	}
}
