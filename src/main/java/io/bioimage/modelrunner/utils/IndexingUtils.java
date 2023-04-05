/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
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
