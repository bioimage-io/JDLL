package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

public abstract class AbstractTensorTransformation implements TensorTransformation
{

	private final String name;

	private Mode mode = Mode.FIXED;

	protected AbstractTensorTransformation( final String name )
	{
		this.name = name;
	}

	@Override
	public String getName()
	{
		return name;
	}

	@Override
	public void setMode( final Mode mode )
	{
		this.mode = mode;
	}

	@Override
	public Mode getMode()
	{
		return mode;
	}

	protected < R extends RealType< R > & NativeType< R > > Tensor< FloatType > makeOutput( final Tensor< R > input )
	{
		final ImgFactory< FloatType > factory = Util.getArrayOrCellImgFactory( input.getData(), new FloatType() );
		final Img< FloatType > outputImg = factory.create( input.getData() );
		final Tensor< FloatType > output = Tensor.build( getName() + '_' + input.getName(), input.getAxesOrderString(), outputImg );
		return output;
	}
}
