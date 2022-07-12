package org.bioimageanalysis.icy.deeplearning.utils;

/**
 * Class that converts flat indexes into multidimensional indexes
 * @author Carlos Javier Garcia Lopez de Haro
 */

public class IndexingUtils {
	
	/**
	 * Obtain a flat index position from a multidimensional index position
	 * @param ind
	 * 	the multidimensional indexes
	 * @param size
	 * 	size of the tensor
	 * @return the index of the position as if it was a flat array
	 */
	public static int multidimensionalIntoFlatIndex(int[] ind, int[] size){
		int flat = 0;
		for (int i = 0; i < ind.length; i ++) {
			int inter = ind[i];
			for (int j = i + 1; j < size.length; j ++)
				inter *= size[j];
			flat += inter;
		}
    	return flat;
	}

}
