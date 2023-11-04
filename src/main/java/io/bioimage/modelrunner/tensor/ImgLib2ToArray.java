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
import net.imglib2.blocks.PrimitiveBlocks;
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
    private static byte[] buildInt8(RandomAccessibleInterval<ByteType> tensor)
    {
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< ByteType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final byte[] flatArr = new byte[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		return flatArr;
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
    private static byte[] buildUint8(RandomAccessibleInterval<UnsignedByteType> tensor)
    {
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< UnsignedByteType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final byte[] flatArr = new byte[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		return flatArr;
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
    private static short[] buildInt16(RandomAccessibleInterval<ShortType> tensor)
    {
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< ShortType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final short[] flatArr = new short[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		return flatArr;
		
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
    private static short[] buildUint16(RandomAccessibleInterval<UnsignedShortType> tensor)
    {
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< UnsignedShortType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final short[] flatArr = new short[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		return flatArr;
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
    private static int[] buildInt32(RandomAccessibleInterval<IntType> tensor)
    {
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< IntType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final int[] flatArr = new int[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		return flatArr;
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
    private static int[] buildUint32(RandomAccessibleInterval<UnsignedIntType> tensor)
    {
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< UnsignedIntType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final int[] flatArr = new int[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		return flatArr;
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
    private static long[] buildInt64(RandomAccessibleInterval<LongType> tensor)
    {
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< LongType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final long[] flatArr = new long[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		return flatArr;
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
    private static float[] buildFloat32(RandomAccessibleInterval<FloatType> tensor)
    {
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< FloatType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final float[] flatArr = new float[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		return flatArr;
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
    private static double[] buildFloat64(RandomAccessibleInterval<DoubleType> tensor)
    {
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< DoubleType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final double[] flatArr = new double[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( tensor.minAsLongArray(), flatArr, sArr );
		return flatArr;
    }
}
