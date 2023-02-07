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
package io.bioimage.modelrunner.utils;

/**
 * Class that converts flat indexes into multidimensional indexes
 * 
 * @author Carlos Javier Garcia Lopez de Haro
 */

public class IndexingUtils
{

	/**
	 * Obtain a flat index position from a multidimensional index position
	 * 
	 * @param ind
	 *            the multidimensional indexes
	 * @param size
	 *            size of the tensor
	 * @return the index of the position as if it was a flat array
	 */
	public static int multidimensionalIntoFlatIndex( int[] ind, int[] size )
	{
		int flat = 0;
		for ( int i = 0; i < ind.length; i++ )
		{
			int inter = ind[ i ];
			for ( int j = i + 1; j < size.length; j++ )
				inter *= size[ j ];
			flat += inter;
		}
		return flat;
	}

	/**
	 * Obtain a flat index position from a multidimensional index position
	 * 
	 * @param ind
	 *            the multidimensional indexes
	 * @param size
	 *            size of the tensor
	 * @return the index of the position as if it was a flat array
	 */
	public static int multidimensionalIntoFlatIndex( long[] ind, long[] size )
	{
		int flat = 0;
		for ( int i = 0; i < ind.length; i++ )
		{
			int inter = (int) ind[ i ];
			for ( int j = i + 1; j < size.length; j++ )
				inter *= size[ j ];
			flat += inter;
		}
		return flat;
	}

}
