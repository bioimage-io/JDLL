package org.bioimageanalysis.icy.deeplearning.transformations;

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
	
	public void checkRequiredArgs() {
		if (threshold == null) {
			throw new IllegalArgumentException("Cannot execute Clip BioImage.io transformation because 'threshold' "
					+ "parameter was not set.");
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
