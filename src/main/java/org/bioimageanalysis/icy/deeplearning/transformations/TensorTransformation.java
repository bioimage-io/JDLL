package org.bioimageanalysis.icy.deeplearning.transformations;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;

import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

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
	 * This method will instantiate a new tensor with the same name, axis
	 * ordering and pixel type that of the input, and write the transformation
	 * results in it.
	 *
	 * @param <R>
	 *            the pixel type of the input and output tensors.
	 * @param input
	 *            the input tensor.
	 * @return a new tensor, of the same pixel type than the input tensor.
	 */
	public < R extends RealType< R > & NativeType< R > > Tensor< R > apply( Tensor< R > input );

	/**
	 * Applies this transformation to the specified input tensor, using the
	 * specified pixel type for calculation.
	 * <p>
	 * This method will instantiate a new tensor with the same name, axis
	 * ordering, but with a pixel type specified in argument. It then will write
	 * the transformation results in it.
	 *
	 * @param <T>
	 *            the desired pixel type of the output tensor.
	 * @param <R>
	 *            the pixel type of the input tensor.
	 * @param input
	 *            the input tensor.
	 * @return a new tensor, of specified pixel type.
	 */
	public < T extends RealType< T > & NativeType< T >, R extends RealType< R > & NativeType< R > > Tensor< T > applyOn( Tensor< R > input, T type );

	/**
	 * Applies this transformation to the specified input tensor, and writes the
	 * results in the specified output tensor.
	 * <p>
	 * It is the caller responsibility to ensure that the tensor backend can be
	 * written in at least the same interval than the input tensor backend.
	 * Calculations will be made abiding to both the tensor pixel types.
	 *
	 * @param <Q>
	 *            the pixel type of the output tensor.
	 * @param <R>
	 *            the pixel type of the input tensor.
	 * @param input
	 *            the input tensor.
	 * @param output
	 *            the output tensor.
	 */
	public < R extends RealType< R > & NativeType< R >, Q extends RealType< Q > & NativeType< Q > > void apply( Tensor< R > input, Tensor< Q > output );

	/**
	 * Applies this transformation to the specified input tensor, and overwrites
	 * it with the results.
	 *
	 * @param <R>
	 *            the pixel type of the input tensor.
	 * @param input
	 *            the input tensor.
	 */
	public < R extends RealType< R > & NativeType< R > > void applyInPlace( Tensor< R > input );

	/**
	 * Returns the name of this transformation.
	 *
	 * @return the name of this transformation.
	 */
	public String getName();

}
