package org.bioimageanalysis.icy.deeplearning.transformations;

public class BinarizeTensorTransformation extends AbstractTensorPixelTransformation
{

	public BinarizeTensorTransformation( final double threshold )
	{
		super( "binarize( " + threshold + " )", v -> ( v >= threshold ) ? 1f : 0f );
	}
}
