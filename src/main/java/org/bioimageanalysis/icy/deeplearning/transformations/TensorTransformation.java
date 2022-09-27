package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * Interface for transformation that change the data in a tensor.
 *
 * @author Jean-Yves Tinevez
 *
 */
public interface TensorTransformation
{

	/**
	 * Applies this transformation to the specified input tensor.
	 * <p>
	 * This method will instantiate a new tensor of floats, with the same name,
	 * and axis ordering that of the input, and write the transformation results
	 * in it.
	 *
	 * @param <R>
	 *            the pixel type of the input tensor.
	 * @param input
	 *            the input tensor.
	 * @return a new tensor with <code>float</code> pixels.
	 */
	public < R extends RealType< R > & NativeType< R > > Tensor< FloatType > apply( Tensor< R > input );

	/**
	 * Applies this transformation to the specified input tensor, and overwrites
	 * it with the results. The input tensor must of type <code>float</code>.
	 *
	 * @param input
	 *            the input tensor.
	 */
	public void applyInPlace( Tensor< FloatType > input );

	/**
	 * Returns the name of this transformation.
	 *
	 * @return the name of this transformation.
	 */
	public String getName();

	default void setMode( final String mode )
	{
		for ( final Mode value : Mode.values() )
		{
			if ( value.toString().equalsIgnoreCase( mode ) )
			{
				setMode( value );
				return;
			}
		}
	}

	public void setMode( Mode mode );

	public Mode getMode();

	/**
	 * Tensor transformation modes.
	 */
	public enum Mode
	{

		FIXED( "fixed" ),
		PER_DATASET( "per_dataset" ),
		PER_SAMPLE( "per_sample" );

		private final String name;

		private Mode( final String name )
		{
			this.name = name;
		}

		@Override
		public String toString()
		{
			return name;
		}
	}
}
