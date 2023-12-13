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
package io.bioimage.modelrunner.tensor;

import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
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
import net.imglib2.util.Cast;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

/**
 * Class that maps {@link Tensor} objects to an array of the corresponding primitive type.
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
     * Gets the {@link Tensor} data into a Java flat array of the corresponding 
     * primitive type with C-ordering
     * 
     * @param <T> 
     * 	the type of the {@link Tensor}
     * @param tensor 
     * 	{@link Tensor} to be flattened into a Java array
     * @return a Java flat array of the corresponding primitive type that contains the data of the {@link RandomAccessibleInterval}
     * 	in C-order
     * @throws IllegalArgumentException If the {@link Tensor} type is not supported.
     */
    public static < T extends RealType< T > & NativeType< T > > Object build(Tensor<T> tensor) throws IllegalArgumentException
    {
		if (tensor.isEmpty())
			return new byte[0];
    	return build(tensor.getData());
    }

    /**
     * Gets the {@link RandomAccessibleInterval} data into a Java flat array of the corresponding 
     * primitive type with C-ordering
     * 
     * @param <T> 
     * 	the type of the {@link RandomAccessibleInterval}
     * @param rai 
     * 	{@link RandomAccessibleInterval} to be flattened into a Java array
     * @return a Java flat array of the corresponding primitive type that contains the data of the {@link RandomAccessibleInterval}
     * 	in C-order
     * @throws IllegalArgumentException If the {@link RandomAccessibleInterval} type is not supported.
     */
    public static < T extends RealType< T > & NativeType< T > > Object build(RandomAccessibleInterval<T> rai) throws IllegalArgumentException
    {
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
    		return buildInt8(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedByteType) {
    		return buildUint8(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof ShortType) {
    		return buildInt16(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedShortType) {
    		return buildUint16(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
    		return buildInt32(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedIntType) {
    		return buildUint32(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof LongType) {
    		return buildInt64(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
    		return buildFloat32(Cast.unchecked(rai));
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
    		return buildFloat64(Cast.unchecked(rai));
    	} else {
            throw new IllegalArgumentException("The image has an unsupported type: " + Util.getTypeFromInterval(rai).getClass().toString());
    	}
    }

    /**
     * Gets the {@link ByteType} {@link RandomAccessibleInterval} data into a Java flat byte array with C-ordering
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be flattened into a Java array
     * @return a Java flat byte array that contains the data of the {@link RandomAccessibleInterval} in C-order
     */
    private static byte[] buildInt8(RandomAccessibleInterval<ByteType> tensor)
    {
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final byte[] flatArr = new byte[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<ByteType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().getByte();
		}
		return flatArr;
    }

    /**
     * Gets the {@link UnsignedByteType} {@link RandomAccessibleInterval} data into a Java flat byte array with C-ordering
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be flattened into a Java array
     * @return a Java flat byte array that contains the data of the {@link RandomAccessibleInterval} in C-order
     */
    private static byte[] buildUint8(RandomAccessibleInterval<UnsignedByteType> tensor)
    {
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final byte[] flatArr = new byte[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<UnsignedByteType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().getByte();
		}
		return flatArr;
    }

    /**
     * Gets the {@link ShortType} {@link RandomAccessibleInterval} data into a Java flat short array with C-ordering
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be flattened into a Java array
     * @return a Java flat short array that contains the data of the {@link RandomAccessibleInterval} in C-order
     */
    private static short[] buildInt16(RandomAccessibleInterval<ShortType> tensor)
    {
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final short[] flatArr = new short[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<ShortType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().getShort();
		}
		return flatArr;
    }

    /**
     * Gets the {@link UnsignedShortType} {@link RandomAccessibleInterval} data into a Java flat short array with C-ordering
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be flattened into a Java array
     * @return a Java flat short array that contains the data of the {@link RandomAccessibleInterval} in C-order
     */
    private static short[] buildUint16(RandomAccessibleInterval<UnsignedShortType> tensor)
    {
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final short[] flatArr = new short[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<UnsignedShortType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().getShort();
		}
		return flatArr;
    }

    /**
     * Gets the {@link IntType} {@link RandomAccessibleInterval} data into a Java flat int array with C-ordering
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be flattened into a Java array
     * @return a Java flat int array that contains the data of the {@link RandomAccessibleInterval} in C-order
     */
    private static int[] buildInt32(RandomAccessibleInterval<IntType> tensor)
    {
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final int[] flatArr = new int[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<IntType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().getInt();
		}
		return flatArr;
    }

    /**
     * Gets the {@link UnsignedIntType} {@link RandomAccessibleInterval} data into a Java flat int array with C-ordering
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be flattened into a Java array
     * @return a Java flat int array that contains the data of the {@link RandomAccessibleInterval} in C-order
     */
    private static int[] buildUint32(RandomAccessibleInterval<UnsignedIntType> tensor)
    {
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final int[] flatArr = new int[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<UnsignedIntType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().getInt();
		}
		return flatArr;
    }

    /**
     * Gets the {@link LongType} {@link RandomAccessibleInterval} data into a Java flat long array with C-ordering
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be flattened into a Java array
     * @return a Java flat long array that contains the data of the {@link RandomAccessibleInterval} in C-order
     */
    private static long[] buildInt64(RandomAccessibleInterval<LongType> tensor)
    {
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final long[] flatArr = new long[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<LongType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().get();
		}
		return flatArr;
    }

    /**
     * Gets the {@link FloatType} {@link RandomAccessibleInterval} data into a Java flat float array with C-ordering
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be flattened into a Java array
     * @return a Java flat float array that contains the data of the {@link RandomAccessibleInterval} in C-order
     */
    private static float[] buildFloat32(RandomAccessibleInterval<FloatType> tensor)
    {
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final float[] flatArr = new float[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<FloatType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().get();
		}
		return flatArr;
    }

    /**
     * Gets the {@link DoubleType} {@link RandomAccessibleInterval} data into a Java flat double array with C-ordering
     * 
     * @param tensor 
     * 	{@link RandomAccessibleInterval} to be flattened into a Java array
     * @return a Java flat double array that contains the data of the {@link RandomAccessibleInterval} in C-order
     */
    private static double[] buildFloat64(RandomAccessibleInterval<DoubleType> tensor)
    {
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final double[] flatArr = new double[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<DoubleType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().get();
		}
		return flatArr;
    }
}
