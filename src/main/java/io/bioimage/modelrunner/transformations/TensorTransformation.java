/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the BioImage.io nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
package io.bioimage.modelrunner.transformations;

import io.bioimage.modelrunner.tensor.Tensor;

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
