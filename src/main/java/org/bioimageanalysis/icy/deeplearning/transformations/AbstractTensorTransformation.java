package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
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
	public < R extends RealType< R > & NativeType< R > > Tensor< R > apply( final Tensor< R > input )
	{
		final R type = Util.getTypeFromInterval( input.getData() );
		return applyOn( input, type );
	}

	@Override
	public < T extends RealType< T > & NativeType< T >, R extends RealType< R > & NativeType< R > > Tensor< T > applyOn( final Tensor< R > input, final T type )
	{
		final ImgFactory< T > factory = Util.getArrayOrCellImgFactory( input.getData(), type );
		final Img< T > outputImg = factory.create( input.getData() );
		final Tensor< T > output = Tensor.build( getName() + '_' + input.getName(), input.getAxesOrderString(), outputImg );
		apply( input, output );
		return output;
	}

	@Override
	public < R extends RealType< R > & NativeType< R > > void applyInPlace( final Tensor< R > input )
	{
		final Tensor< R > tmp = apply( input );
		// Copy tmp results back to input.
		LoopBuilder.setImages( tmp.getData(), input.getData() )
			.multiThreaded()
			.forEachPixel( ( i, o ) -> o.setReal( i.getRealDouble() ) );
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
}
