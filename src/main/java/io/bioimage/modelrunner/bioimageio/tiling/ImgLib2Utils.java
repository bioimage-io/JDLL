/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.bioimageio.tiling;

import java.util.stream.LongStream;


import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import net.imglib2.loops.LoopBuilder;

/**
 * Class to handle {@link RandomAccessibleInterval} operations, as mirroring or tiling
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando 
 * 
 */

public final class ImgLib2Utils
{

    private ImgLib2Utils()
    {
    }

    /**
     * Copies values from source to target NDArrays starting at the source offset position as the origin in target NDArrays and taking the given size. This
     * method assumes the target NDArray is already initialized with all the needed space.
     * 
     * @param <T>
     * 	possible data types the ImgLib2 object might have
     * @param sourceNDArray
     *        Source data INDArray.
     * @param targetNDArray
     *        Target INDArray.
     * @param sourceOffset
     *        Position in source sequence to start the copy from. Must be of length 5.
     * @param paddingBottomRight
     *        the size of the padding at the bottom or right sides for the special case
     *        of the last patch at which some extra memory can be saved
     */
    public static < T extends RealType< T > & NativeType< T > > void copyRaiData(RandomAccessibleInterval<T> sourceNDArray, RandomAccessibleInterval<T> targetNDArray, int[] sourceOffset,
            int[] paddingBottomRight) {
        long[] sourceSize = sourceNDArray.dimensionsAsLongArray();
        long[] targetSize = targetNDArray.dimensionsAsLongArray();
        // Recalculate the source offset for the case it is the last patch,
        // so no extra memory is used during memory and only the halo has to be
        // mirrored
        long[] newSourceOffset = LongStream.range(0, sourceOffset.length)
        		.map(i -> Math.min(sourceOffset[(int) i], sourceSize[(int) i] + paddingBottomRight[(int) i] - targetSize[(int) i]))
        		.toArray();
        long[] internalSourceOffset = LongStream.range(0, sourceSize.length)
                .map(i -> Math.min(sourceSize[(int) i], Math.max(0, newSourceOffset[(int) i])))
                .toArray();
        long[] internalSize = LongStream.range(0, sourceSize.length)
                .map(i -> Math.max(0,
                        Math.min(newSourceOffset[(int) i] + targetSize[(int) i], sourceSize[(int) i]) - internalSourceOffset[(int) i]))
                .toArray();
        long[] targetRaiStart = LongStream.range(0, sourceSize.length)
        			.map(i -> internalSourceOffset[(int) i] - newSourceOffset[(int) i]).toArray();
        IntervalView<T> sourceTile = Views.offsetInterval( sourceNDArray, internalSourceOffset, internalSize );
        IntervalView<T> targetTile = Views.offsetInterval( targetNDArray, targetRaiStart, internalSize );
        LoopBuilder.setImages( sourceTile, targetTile )
				.multiThreaded()
				.forEachPixel( (i, j) -> j.set( i ) );
    }

    /**
     * Takes the {@code INDArray} and adds a mirror on the areas out of the patch (specified by patchStart and patchSize). Boundaries are checked with both
     * the input and the patch sequences. The result is added to the {@code patchSequence}.
     * 
     * @param <T>
     * 	ImgLib2 data types the image might have
     * @param inputNDArr
     *        The INDArray the patch was taken from.
     * @param patchNDArr
     *        The patch extracted from the input INDArray.
     * @param newPatchStart
     *        The patch position in the input INDArray. First array start of the 
     *        front padding and second array, start of the back.
     * @param padding
     *        the padding of the image
     */
    public static < T extends RealType< T > & NativeType< T > > void addMirrorToPatchRai(RandomAccessibleInterval<T> inputNDArr, RandomAccessibleInterval<T> patchNDArr,
            int[] newPatchStart, int[][] padding) {
    	long[] patchSize = patchNDArr.dimensionsAsLongArray();
    	long[] imSize = inputNDArr.dimensionsAsLongArray();
    	// Recalculate when the patch starts w.r.t. the source image for the case it is the last patch,
        // so no extra memory is used during memory and only the halo has to be
        // mirrored
        long[] patchStart = LongStream.range(0, newPatchStart.length)
        		.map(i -> Math.min(newPatchStart[(int) i], imSize[(int) i] + padding[1][(int) i] - patchSize[(int) i]))
        		.toArray();
        long[] paddingFront = LongStream.range(0, patchStart.length)
        						.map(i -> patchStart[(int) i] < 0 ? patchStart[(int) i] * -1 : 0).toArray();
        // TODO check well the padding front and expand mirror parameters
        patchNDArr = Views.offsetInterval( Views.expandMirrorDouble(inputNDArr, paddingFront), 
        		patchStart, patchNDArr.dimensionsAsLongArray());
        /*
        Sequence seq = ImgLib2ToSequence.build(patchNDArr, "yxc");
        seq.setName("mirror_input_path");
        Icy.getMainInterface().addSequence(seq);
        */
        
    }

    /**
     * Uses the {@code patchSequence} to fill the {@code resultSequence} at the {@code resultOffset} position. The copied interval in the patch is given by the
     * {@code patchOffset} and the {@code patchSize}.
     * 
     * @param <T>
     * 	possible ImgLib2 data types the {@link RandomAccessibleInterval} might have
     * @param resultNDArray
     *        The sequence where the data is pasted at.
     * @param patchNDArray
     *        The sequence where the data is copied from.
     * @param resultOffset
     *        The position in the result sequence where the data is pasted.
     * @param paddingFront
     *        The position in the patch sequence where the copy starts.
     * @param areaOfInterestSize
     *        The length on each axis to be copied from the patch sequence.
     *        This size only corresponds to the area of interest, the pixels
     *        corresponding to padding are left out
     */
    public static < T extends RealType< T > & NativeType< T > > void fillRaiAt(RandomAccessibleInterval<T> resultNDArray, 
    		RandomAccessibleInterval<T> patchNDArray, int[] resultOffset,
            int[] paddingFront, int[] areaOfInterestSize)
    {
        long[] finalSize = resultNDArray.dimensionsAsLongArray();
    	// Recalculate the source offset for the case it is the last patch,
        // so no extra memory is used during memory and only the halo has to be
        // mirrored
        long[] newPaddingFront = LongStream.range(0, resultOffset.length)
        		.map(i -> Math.min(resultOffset[(int) i], finalSize[(int) i] - areaOfInterestSize[(int) i]))
        		.toArray();        
        long[] startSource = LongStream.range(0, paddingFront.length)
        		.map(i -> newPaddingFront[(int)i] < 0 ? -1 * newPaddingFront[(int)i] + paddingFront[(int)i]
        												: paddingFront[(int)i]).toArray();
        long[] startTarget = LongStream.range(0, paddingFront.length)
        		.map(i -> Math.max(0, newPaddingFront[(int) i])).toArray();
        long[] endTarget = LongStream.range(0, paddingFront.length)
        		.map(i -> Math.min(finalSize[(int) i], newPaddingFront[(int) i] + areaOfInterestSize[(int) i]))
        		.toArray();
        long[] intervalSize = LongStream.range(0, paddingFront.length)
        		.map(i -> endTarget[(int) i] - startTarget[(int) i]).toArray();
        IntervalView<T> resultInterval = Views.offsetInterval(resultNDArray, startTarget, intervalSize);
        IntervalView<T> patchInterval = Views.offsetInterval(patchNDArray, startSource, intervalSize);
        LoopBuilder.setImages( patchInterval, resultInterval )
				.multiThreaded()
				.forEachPixel( (i, j) -> j.set( i ));
        /*
        Sequence seq = ImgLib2ToSequence.build(patchNDArray, "byxc");
        seq.setName("output_result");
        Icy.getMainInterface().addSequence(seq);
        */
    }
	
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
	
	/**
	 * Create a copy of the int array but with 5 dimension. The difference in dimensions
	 * is compensated by 0s
	 * @param arr
	 * 	the array to be extended
	 * @return array extended with zeros
	 */
	public static int[] to5DFillWith0(int[] arr) {
		int[] nArr = new int[5];
		for (int i = 0; i < arr.length; i ++)
			nArr[i] = arr[i];
		return nArr;
	}
	
	/**
	 * Create a copy of the int array but with 5 dimension. The difference in dimensions
	 * is compensated by 1s
	 * @param arr
	 * 	the array to be extended
	 * @return array extended with ones
	 */
	public static int[] to5DFillWith1(int[] arr) {
		int[] nArr = new int[] {1, 1, 1, 1, 1};
		for (int i = 0; i < arr.length; i ++)
			nArr[i] = arr[i];
		return nArr;
	}

    /**
     * Creates an array containing the size of the given {@link RandomAccessibleInterval}.
     * It is the same as teh shape
     * 
	 * @param <T>
	 * 	Possible input ImgLib2 data types 
     * @param s
     * 	Target NDArray
     * @return int array with the shape of the NDArray
     */
    public static < T extends RealType< T > & NativeType< T > > int[] getRaiSizeArray(RandomAccessibleInterval<T> s)
    {
    	long[] longShape = s.dimensionsAsLongArray();
    	int nDims = longShape.length;
    	int[] intShape = new int[longShape.length];
    	for (int i = 0; i < nDims; i ++)
    		intShape[i] = (int) longShape[i];
        return intShape;
    }
}
