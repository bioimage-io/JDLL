package org.bioimageanalysis.icy.deeplearning.transformations;

public class SigmoidTensorTransformation extends AbstractTensorPixelTransformation
{

	public SigmoidTensorTransformation()
	{
		super(
				"sigmoid",
				v -> ( float ) ( 1. / ( 1. + Math.exp( -v ) ) ) );
	}
}
