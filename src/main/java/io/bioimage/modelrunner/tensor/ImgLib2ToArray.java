/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java 0.3.0 and newer API for Tensorflow 2.
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
package io.bioimage.modelrunner.tensor;

import java.nio.ByteBuffer;

import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;

/**
 * Class that maps {@link Tensor} objects to {@link ByteBuffer} objects.
 * This is done to modify the files that are used to communicate between process
 * to avoid the TF2-TF1/Pytorch incompatibility that happens in these systems
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class ImgLib2ToArray
{
    /**
     * Not used (Utility class).
     */
    private ImgLib2ToArray()
    {
    }

    /**
     * Maps a {@link Tensor} to the provided {@link ByteBuffer} with all the information
     * needed to reconstruct the tensor again
     * 
     * @param <T> 
     * 	the type of the tensor
     * @param tensor 
     * 	tensor to be mapped into byte buffer
     * @param byteBuffer 
     * 	target byte bufer
     * @throws IllegalArgumentException
     *         If the {@link Tensor} ImgLib2 type is not supported.
     */
    public static < T extends RealType< T > & NativeType< T > > Object build(Tensor<T> tensor)
    {
		if (tensor.isEmpty())
			return new byte[0];
    	return build(tensor.getData());
    }

    /**
     * Adds the {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param <T> 
     * 	the type of the {@link RandomAccessibleInterval}
     * @param rai 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     * @throws IllegalArgumentException If the {@link RandomAccessibleInterval} type is not supported.
     */
    public static <T extends Type<T>> Object build(RandomAccessibleInterval<T> rai)
    {
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
    		return buildInt8((RandomAccessibleInterval<ByteType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedByteType) {
    		return buildUint8((RandomAccessibleInterval<UnsignedByteType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof ShortType) {
    		return buildInt16((RandomAccessibleInterval<ShortType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedShortType) {
    		return buildUint16((RandomAccessibleInterval<UnsignedShortType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
    		return buildInt32((RandomAccessibleInterval<IntType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedIntType) {
    		return buildUint32((RandomAccessibleInterval<UnsignedIntType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof LongType) {
    		return buildInt64((RandomAccessibleInterval<LongType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
    		return buildFloat32((RandomAccessibleInterval<FloatType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
    		return buildFloat64((RandomAccessibleInterval<DoubleType>) rai);
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
    private static byte[] buildInt8(RandomAccessibleInterval<ByteType> imgTensor)
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
     * Adds the ByteType {@link RandomAccessibleInterval} data to the {@link ByteBuffer} provided.
     * The position of the ByteBuffer is kept in the same place as it was received.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param byteBuffer 
     * 	target bytebuffer
     */
    private static short[] buildUint8(RandomAccessibleInterval<UnsignedByteType> imgTensor)
    {
    	Cursor<UnsignedByteType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<UnsignedByteType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<UnsignedByteType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long ss : imgTensor.dimensionsAsLongArray()) {flatSize *= ss;}
		short[] byteArr = new short[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			byteArr[cc ++] = tensorCursor.get().getByte();
		}
		return byteArr;
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
    private static short[] buildInt16(RandomAccessibleInterval<ShortType> imgTensor)
    {
    	Cursor<ShortType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<ShortType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<ShortType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long ss : imgTensor.dimensionsAsLongArray()) {flatSize *= ss;}
		short[] byteArr = new short[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			byteArr[cc ++] = tensorCursor.get().get();
		}
		return byteArr;
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
    private static int[] buildUint16(RandomAccessibleInterval<UnsignedShortType> imgTensor)
    {
    	Cursor<UnsignedShortType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<UnsignedShortType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<UnsignedShortType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long ss : imgTensor.dimensionsAsLongArray()) {flatSize *= ss;}
		int[] byteArr = new int[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			byteArr[cc ++] = tensorCursor.get().get();
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
    private static int[] buildInt32(RandomAccessibleInterval<IntType> imgTensor)
    {
    	Cursor<IntType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<IntType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<IntType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long ss : imgTensor.dimensionsAsLongArray()) {flatSize *= ss;}
		int[] byteArr = new int[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			byteArr[cc ++] = tensorCursor.get().getInt();
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
    private static long[] buildUint32(RandomAccessibleInterval<UnsignedIntType> imgTensor)
    {
    	Cursor<UnsignedIntType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<UnsignedIntType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<UnsignedIntType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long ss : imgTensor.dimensionsAsLongArray()) {flatSize *= ss;}
		long[] byteArr = new long[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			byteArr[cc ++] = tensorCursor.get().get();
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
    private static long[] buildInt64(RandomAccessibleInterval<LongType> imgTensor)
    {
    	Cursor<LongType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<LongType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<LongType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long ss : imgTensor.dimensionsAsLongArray()) {flatSize *= ss;}
		long[] byteArr = new long[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			byteArr[cc ++] = tensorCursor.get().get();
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
    private static float[] buildFloat32(RandomAccessibleInterval<FloatType> imgTensor)
    {
    	Cursor<FloatType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<FloatType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<FloatType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long ss : imgTensor.dimensionsAsLongArray()) {flatSize *= ss;}
		float[] byteArr = new float[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			byteArr[cc ++] = tensorCursor.get().getRealFloat();
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
    private static double[] buildFloat64(RandomAccessibleInterval<DoubleType> imgTensor)
    {
    	Cursor<DoubleType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<DoubleType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<DoubleType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long ss : imgTensor.dimensionsAsLongArray()) {flatSize *= ss;}
		double[] byteArr = new double[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			byteArr[cc ++] = tensorCursor.get().getRealDouble();
		}
		return byteArr;
    }
}
