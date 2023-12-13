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
package io.bioimage.modelrunner.numpy;

import java.nio.ByteBuffer;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;

public class DecodeImgLib2 {

	
	public static  < T extends RealType< T > & NativeType< T > >
				byte[] tensorDataToByteArray(Tensor<T> tensor) {
		return imglib2ToByteArray(tensor.getData());
	}

	
	public static < T extends RealType< T > & NativeType< T > >
				byte[] imglib2ToByteArray(RandomAccessibleInterval<T> rai) {
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
    		return Cast.unchecked(buildByte(Cast.unchecked(rai)));
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
    		return Cast.unchecked(buildInt(Cast.unchecked(rai)));
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
    		return Cast.unchecked(buildFloat(Cast.unchecked(rai)));
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
    		return Cast.unchecked(buildDouble(Cast.unchecked(rai)));
    	} else {
            throw new IllegalArgumentException("The image has an unsupported type: " + Util.getTypeFromInterval(rai).getClass().toString());
    	}
	}

    /**
     * Adds the ByteType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private static byte[] buildByte(RandomAccessibleInterval<ByteType> imgTensor)
    {
    	Cursor<ByteType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<ByteType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<ByteType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long ss : imgTensor.dimensionsAsLongArray()) {flatSize *= ss;}
		byte[] byteArr = new byte[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			byteArr[cc ++] = tensorCursor.get().getByte();
		}
		return byteArr;
    }

    /**
     * Adds the IntType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private static byte[] buildInt(RandomAccessibleInterval<IntType> imgTensor)
    {
    	Cursor<IntType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<IntType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<IntType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 4;
		for (long ss : imgTensor.dimensionsAsLongArray()) {flatSize *= ss;}
		byte[] byteArr = new byte[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int val = tensorCursor.get().getInt();
			byte[] arr = ByteBuffer.allocate(4).putInt(val).array();
			System.arraycopy(arr, 0, byteArr, cc * 4, 4);
			cc ++;
		}
		return byteArr;
    }

    /**
     * Adds the FloatType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private static byte[] buildFloat(RandomAccessibleInterval<FloatType> imgTensor)
    {
    	Cursor<FloatType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<FloatType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<FloatType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 4;
		for (long ss : imgTensor.dimensionsAsLongArray()) {flatSize *= ss;}
		byte[] byteArr = new byte[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			float val = tensorCursor.get().getRealFloat();
			byte[] arr = ByteBuffer.allocate(4).putFloat(val).array();;
			System.arraycopy(arr, 0, byteArr, cc * 4, 4);
			cc ++;
		}
		return byteArr;
    }

    /**
     * Adds the DoubleType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private static byte[] buildDouble(RandomAccessibleInterval<DoubleType> imgTensor)
    {
    	Cursor<DoubleType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<DoubleType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<DoubleType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 8;
		for (long ss : imgTensor.dimensionsAsLongArray()) {flatSize *= ss;}
		byte[] byteArr = new byte[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			double val = tensorCursor.get().getRealDouble();
			byte[] arr = ByteBuffer.allocate(4).putDouble(val).array();;
			System.arraycopy(arr, 0, byteArr, cc * 8, 8);
			cc ++;
		}
		return byteArr;
    }
}
