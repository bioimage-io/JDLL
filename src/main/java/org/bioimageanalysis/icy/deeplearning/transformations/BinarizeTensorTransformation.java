package org.bioimageanalysis.icy.deeplearning.transformations;

import java.util.ArrayList;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class BinarizeTensorTransformation extends AbstractTensorPixelTransformation
{
	
	private static String name = "binarize";
	private Double threshold;

	public BinarizeTensorTransformation()
	{
		super( name );
	}
	
	public void setThreshold(Object threshold) {
		if (threshold instanceof Integer) {
			this.threshold = Double.valueOf((int) threshold);
		} else if (threshold instanceof Double) {
			this.threshold = (double) threshold;
		} else if (threshold instanceof String) {
			this.threshold = Double.valueOf((String) threshold);
		} else {
			throw new IllegalArgumentException("'threshold' parameter has to be either and instance of "
					+ Integer.class + " or " + Double.class
					+ ". The provided argument is an instance of: " + threshold.getClass());
		}
	}
	
	public void checkRequiredArgs() {
		if (threshold == null) {
			throw new IllegalArgumentException(String.format(DEFAULT_MISSING_ARG_ERR, "threshold"));
		}
	}

	public < R extends RealType< R > & NativeType< R > > Tensor< FloatType > apply( final Tensor< R > input )
	{
		checkRequiredArgs();
		super.setFloatUnitaryOperator(v -> ( v >= threshold ) ? 1f : 0f);
		return super.apply(input);
	}

	public void applyInPlace( final Tensor< FloatType > input )
	{
		checkRequiredArgs();
		super.setFloatUnitaryOperator(v -> ( v >= threshold ) ? 1f : 0f);
		super.apply(input);
	}
}
