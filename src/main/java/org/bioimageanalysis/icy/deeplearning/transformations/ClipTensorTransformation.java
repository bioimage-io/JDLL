package org.bioimageanalysis.icy.deeplearning.transformations;

public class ClipTensorTransformation extends AbstractTensorPixelTransformation
{

	private static final class ClipFunction implements FloatUnaryOperator
	{

		private final float min;

		private final float max;

		private ClipFunction( final double min, final double max )
		{
			this.min = ( float ) Math.min( min, max );
			this.max = ( float ) Math.max( min, max );
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

	public ClipTensorTransformation( final double min, final double max )
	{
		super(
				"clip( " + Math.min( min, max ) + ", " + Math.max( min, max ) + " )",
				new ClipFunction( min, max ) );
	}
}
