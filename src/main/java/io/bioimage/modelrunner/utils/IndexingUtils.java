/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
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
	 * Obtain a flat index position from a multidimensional index position, assumes row major (C-order)
	 *
	 * @param ind the ind.
	 * @param size the size.
	 * @return the resulting int.
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
	 * Executes main.
	 *
	 * @param args the args parameter.
	 */
	public static void main(String[] args) {
		int[] size = new int[] {3, 3, 2};
		int[] pos0 = new int[] {0, 0, 0};
		int[] pos1 = new int[] {0, 0, 1};
		int[] pos2 = new int[] {0, 2, 1};
		int[] pos3 = new int[] {1, 0, 0};
		int[] posn = new int[] {1, 100, 7};
		int[] sizen = new int[] {3, 256, 15};
		System.out.println(multidimensionalIntoFlatIndex(pos0, size));
		System.out.println(multidimensionalIntoFlatIndex(pos1, size));
		System.out.println(multidimensionalIntoFlatIndex(pos2, size));
		System.out.println(multidimensionalIntoFlatIndex(pos3, size));
		System.out.println(multidimensionalIntoFlatIndex(posn, sizen));
		System.out.println(Arrays.toString(flatIntoMultidimensionalIndex(multidimensionalIntoFlatIndex(pos0, size), size)));
		System.out.println(Arrays.toString(flatIntoMultidimensionalIndex(multidimensionalIntoFlatIndex(pos1, size), size)));
		System.out.println(Arrays.toString(flatIntoMultidimensionalIndex(multidimensionalIntoFlatIndex(pos2, size), size)));
		System.out.println(Arrays.toString(flatIntoMultidimensionalIndex(multidimensionalIntoFlatIndex(pos3, size), size)));
		System.out.println(Arrays.toString(flatIntoMultidimensionalIndex(multidimensionalIntoFlatIndex(posn, sizen), sizen)));
	}

	/**
	 * Obtain a flat index position from a multidimensional index position, assumes row major (C-order)
	 *
	 * @param ind the ind.
	 * @param size the size.
	 * @return the resulting int.
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
	 * Obtain the multidimensional position corresponding to a flat position in an nd array
	 *
	 * @param flat the flat.
	 * @param size the size.
	 * @return the resulting long.
	 */
	public static long[] flatIntoMultidimensionalIndex( long flat, long[] size )
	{
		long[] ind = new long[size.length];
		long rem = 0 + flat;
		for ( int i = 0; i < ind.length; i++ )
		{
			int inter = 1;
			for ( int j = i + 1; j < size.length; j++ )
				inter *= size[ j ];
			ind[i] = rem / inter;
			rem = rem % inter;
		}
		return ind;
	}

	/**
	 * Obtain the multidimensional position corresponding to a flat position in an nd array
	 *
	 * @param flat the flat.
	 * @param size the size.
	 * @return the resulting int.
	 */
	public static int[] flatIntoMultidimensionalIndex( int flat, int[] size )
	{
		int[] ind = new int[size.length];
		int rem = 0 + flat;
		for ( int i = 0; i < ind.length; i++ )
		{
			int inter = 1;
			for ( int j = i + 1; j < size.length; j++ )
				inter *= size[ j ];
			ind[i] = rem / inter;
			rem = rem % inter;
		}
		return ind;
	}

	/**
	 * Obtain the multidimensional position corresponding to a flat position using
	 * a reference axes order for the flat iteration.
	 * <p>
	 * The returned index follows {@code axesOrder}, but the flat index is decoded
	 * as if the grid were ordered by {@code referenceAxesOrder}. This keeps tiled
	 * input and output tensors aligned when their tensor axes differ, for example
	 * input {@code xycb} and output {@code byx}.
	 *
	 * @param flat the flat index.
	 * @param size the grid size following {@code axesOrder}.
	 * @param axesOrder the axes order of {@code size} and of the returned index.
	 * @param referenceAxesOrder the semantic axes order used to decode {@code flat}.
	 * @return the multidimensional index following {@code axesOrder}.
	 */
	public static int[] flatIntoMultidimensionalIndex( int flat, int[] size,
			String axesOrder, String referenceAxesOrder )
	{
		axesOrder = normalizeAxesOrder(axesOrder, "axesOrder");
		referenceAxesOrder = normalizeAxesOrder(referenceAxesOrder, "referenceAxesOrder");
		if (size.length != axesOrder.length()) {
			throw new IllegalArgumentException("Size length (" + size.length
					+ ") does not match axes order '" + axesOrder + "'.");
		}
		if (axesOrder.equals(referenceAxesOrder)) {
			return flatIntoMultidimensionalIndex(flat, size);
		}
		String referenceOrder = commonReferenceOrder(axesOrder, referenceAxesOrder);
		int[] referenceSize = new int[referenceOrder.length()];
		for (int i = 0; i < referenceOrder.length(); i ++) {
			referenceSize[i] = size[axesOrder.indexOf(referenceOrder.charAt(i))];
		}
		return flatIntoMultidimensionalIndex(flat, size, axesOrder, referenceOrder, referenceSize);
	}

	/**
	 * Obtain the multidimensional position corresponding to a flat position using
	 * a full reference grid for the flat iteration.
	 * <p>
	 * Missing local axes are ignored, so tensors without a reference axis are
	 * reused across that axis. Local axes with grid size {@code 1} are also reused
	 * across larger reference grids. Any other grid mismatch is rejected because
	 * there is no unambiguous tile-to-tile mapping.
	 *
	 * @param flat the flat index.
	 * @param size the local grid size following {@code axesOrder}.
	 * @param axesOrder the axes order of {@code size} and of the returned index.
	 * @param referenceAxesOrder the axes order of {@code referenceSize}.
	 * @param referenceSize the grid size used to decode {@code flat}.
	 * @return the multidimensional index following {@code axesOrder}.
	 */
	public static int[] flatIntoMultidimensionalIndex( int flat, int[] size,
			String axesOrder, String referenceAxesOrder, int[] referenceSize )
	{
		axesOrder = normalizeAxesOrder(axesOrder, "axesOrder");
		referenceAxesOrder = normalizeAxesOrder(referenceAxesOrder, "referenceAxesOrder");
		if (size.length != axesOrder.length()) {
			throw new IllegalArgumentException("Size length (" + size.length
					+ ") does not match axes order '" + axesOrder + "'.");
		}
		if (referenceSize == null || referenceSize.length != referenceAxesOrder.length()) {
			throw new IllegalArgumentException("Reference size length does not match axes order '"
					+ referenceAxesOrder + "'.");
		}
		int[] referenceIndex = flatIntoMultidimensionalIndex(flat, referenceSize);
		int[] index = new int[size.length];
		for (int i = 0; i < axesOrder.length(); i ++) {
			char axis = axesOrder.charAt(i);
			int referenceAxisIndex = referenceAxesOrder.indexOf(axis);
			if (referenceAxisIndex < 0) {
				if (size[i] != 1) {
					throw new IllegalArgumentException("Axis '" + axis + "' is tiled locally but absent from "
							+ "reference axes order '" + referenceAxesOrder + "'.");
				}
				index[i] = 0;
				continue;
			}
			int referenceAxisPosition = referenceIndex[referenceAxisIndex];
			if (size[i] == 1) {
				index[i] = 0;
			} else if (referenceSize[referenceAxisIndex] != size[i]) {
				throw new IllegalArgumentException("Cannot map tiled axis '" + axis + "' from reference grid "
						+ referenceSize[referenceAxisIndex] + " to local grid " + size[i] + ".");
			} else {
				index[i] = referenceAxisPosition;
			}
		}
		return index;
	}

	private static String commonReferenceOrder(String axesOrder, String referenceAxesOrder) {
		StringBuilder order = new StringBuilder();
		for (int i = 0; i < referenceAxesOrder.length(); i ++) {
			char axis = referenceAxesOrder.charAt(i);
			if (axesOrder.indexOf(axis) >= 0 && order.indexOf(String.valueOf(axis)) < 0) {
				order.append(axis);
			}
		}
		for (int i = 0; i < axesOrder.length(); i ++) {
			char axis = axesOrder.charAt(i);
			if (order.indexOf(String.valueOf(axis)) < 0) {
				order.append(axis);
			}
		}
		return order.toString();
	}

	private static String normalizeAxesOrder(String axesOrder, String name) {
		if (axesOrder == null || axesOrder.trim().isEmpty()) {
			throw new IllegalArgumentException(name + " cannot be null or empty.");
		}
		axesOrder = axesOrder.toLowerCase().replace(" ", "");
		for (int i = 0; i < axesOrder.length(); i ++) {
			char axis = axesOrder.charAt(i);
			if (!Character.isLetter(axis)) {
				throw new IllegalArgumentException(name + " must contain only letters, got '" + axesOrder + "'.");
			}
			if (axesOrder.indexOf(axis) != axesOrder.lastIndexOf(axis)) {
				throw new IllegalArgumentException(name + " contains repeated axis '" + axis + "': '" + axesOrder + "'.");
			}
		}
		return axesOrder;
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
