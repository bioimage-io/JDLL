/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.utils;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

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
	
	public static void main(String[] args) {
		int[] size = new int[] {3, 3, 2};
		int[] pos0 = new int[] {0, 0, 0};
		int[] pos1 = new int[] {0, 0, 1};
		int[] pos2 = new int[] {0, 0, 2};
		int[] pos3 = new int[] {1, 0, 0};
		System.out.println(multidimensionalIntoFlatIndex(pos0, size));
		System.out.println(multidimensionalIntoFlatIndex(pos1, size));
		System.out.println(multidimensionalIntoFlatIndex(pos2, size));
		System.out.println(multidimensionalIntoFlatIndex(pos3, size));
		System.out.println(multidimensionalIntoFlatIndex(pos0, size));
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
	
    /**
     * Argsort method
     *
     * @param list list of integers
     * @return the order of the list of integers
     */
    public static Integer[] argsort(List<Integer> list) {
        Integer[] indices = new Integer[list.size()];
        for (int i = 0; i < list.size(); i++) {
            indices[i] = i;
        }
        indices = indices.clone(); // Clone the array since Arrays.sort modifies it
        Arrays.sort(indices, Comparator.comparingInt(list::get));
        return indices;
    }

}
