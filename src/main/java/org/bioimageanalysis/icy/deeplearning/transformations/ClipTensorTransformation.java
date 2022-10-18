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
	
	public void setMin(Object min) {
		if (min instanceof Integer) {
			this.min = Double.valueOf((int) min);
		} else if (min instanceof Double) {
			this.min = (double) min;
		} else if (min instanceof String) {
			this.min = Double.valueOf((String) min);
		} else {
			throw new IllegalArgumentException("'min' parameter has to be either and instance of "
					+ Integer.class + " or " + Double.class
					+ ". The provided argument is an instance of: " + min.getClass());
		}
	}
	
	public void setMax(Object max) {
		if (max instanceof Integer) {
			this.max = Double.valueOf((int) max);
		} else if (max instanceof Double) {
			this.max = (double) max;
		} else if (max instanceof String) {
			this.max = Double.valueOf((String) max);
		} else {
			throw new IllegalArgumentException("'max' parameter has to be either and instance of "
					+ Integer.class + " or " + Double.class
					+ ". The provided argument is an instance of: " + max.getClass());
		}
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
