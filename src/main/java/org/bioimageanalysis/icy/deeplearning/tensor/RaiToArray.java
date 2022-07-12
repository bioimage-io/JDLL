package org.bioimageanalysis.icy.deeplearning.tensor;

import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.ByteArray;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.basictypeaccess.array.IntArray;
import net.imglib2.img.basictypeaccess.array.LongArray;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;

/**
 * Class to copy RandomAccessibleIntervals into Java Arrays
 * Check for possible problems at: 
 * https://github.com/imagej/imagej-tensorflow/blob/2f422ad7955075bbf94d20536ee3904b69cd4926/src/main/java/net/imagej/tensorflow/Tensors.java#L1020
 * 
 * 
 * @author Carlos Javier Garcia Lopez de Haro
 *
 */

public class RaiToArray {


	public static byte[] byteArray(
		final RandomAccessibleInterval<ByteType> image)
	{
		final byte[] array = extractByteArray(image);
		return array == null ? createByteArray(image) : array;
	}

	public static double[] doubleArray(
		final RandomAccessibleInterval<DoubleType> image)
	{
		final double[] array = extractDoubleArray(image);
		return array == null ? createDoubleArray(image) : array;
	}

	public static float[] floatArray(
		final RandomAccessibleInterval<FloatType> image)
	{
		final float[] array = extractFloatArray(image);
		return array == null ? createFloatArray(image) : array;
	}

	public static int[] intArray(
		final RandomAccessibleInterval<IntType> image)
	{
		final int[] array = extractIntArray(image);
		return array == null ? createIntArray(image) : array;
	}

	public static long[] longArray(
		final RandomAccessibleInterval<LongType> image)
	{
		final long[] array = extractLongArray(image);
		return array == null ? createLongArray(image) : array;
	}

	private static byte[] createByteArray(
		final RandomAccessibleInterval<ByteType> image)
	{
		final long[] dims = Intervals.dimensionsAsLongArray(image);
		final ArrayImg<ByteType, ByteArray> dest = ArrayImgs.bytes(dims);
		copy(image, dest);
		return dest.update(null).getCurrentStorageArray();
	}

	private static double[] createDoubleArray(
		final RandomAccessibleInterval<DoubleType> image)
	{
		final long[] dims = Intervals.dimensionsAsLongArray(image);
		final ArrayImg<DoubleType, DoubleArray> dest = ArrayImgs.doubles(dims);
		copy(image, dest);
		return dest.update(null).getCurrentStorageArray();
	}

	private static float[] createFloatArray(
		final RandomAccessibleInterval<FloatType> image)
	{
		final long[] dims = Intervals.dimensionsAsLongArray(image);
		final ArrayImg<FloatType, FloatArray> dest = ArrayImgs.floats(dims);
		copy(image, dest);
		return dest.update(null).getCurrentStorageArray();
	}

	private static int[] createIntArray(
		final RandomAccessibleInterval<IntType> image)
	{
		final long[] dims = Intervals.dimensionsAsLongArray(image);
		final ArrayImg<IntType, IntArray> dest = ArrayImgs.ints(dims);
		copy(image, dest);
		return dest.update(null).getCurrentStorageArray();
	}

	private static long[] createLongArray(
		final RandomAccessibleInterval<LongType> image)
	{
		final long[] dims = Intervals.dimensionsAsLongArray(image);
		final ArrayImg<LongType, LongArray> dest = ArrayImgs.longs(dims);
		copy(image, dest);
		return dest.update(null).getCurrentStorageArray();
	}

	private static byte[] extractByteArray(
		final RandomAccessibleInterval<ByteType> image)
	{
		if (!(image instanceof ArrayImg)) return null;
		@SuppressWarnings("unchecked")
		final ArrayImg<ByteType, ?> arrayImg = (ArrayImg<ByteType, ?>) image;
		final Object dataAccess = arrayImg.update(null);
		return dataAccess instanceof ByteArray ? //
			((ByteArray) dataAccess).getCurrentStorageArray() : null;
	}

	private static double[] extractDoubleArray(
		final RandomAccessibleInterval<DoubleType> image)
	{
		if (!(image instanceof ArrayImg)) return null;
		@SuppressWarnings("unchecked")
		final ArrayImg<DoubleType, ?> arrayImg = (ArrayImg<DoubleType, ?>) image;
		final Object dataAccess = arrayImg.update(null);
		return dataAccess instanceof DoubleArray ? //
			((DoubleArray) dataAccess).getCurrentStorageArray() : null;
	}

	private static float[] extractFloatArray(
		final RandomAccessibleInterval<FloatType> image)
	{
		if (!(image instanceof ArrayImg)) return null;
		@SuppressWarnings("unchecked")
		final ArrayImg<FloatType, ?> arrayImg = (ArrayImg<FloatType, ?>) image;
		final Object dataAccess = arrayImg.update(null);
		return dataAccess instanceof FloatArray ? //
			((FloatArray) dataAccess).getCurrentStorageArray() : null;
	}

	private static int[] extractIntArray(
		final RandomAccessibleInterval<IntType> image)
	{
		if (!(image instanceof ArrayImg)) return null;
		@SuppressWarnings("unchecked")
		final ArrayImg<IntType, ?> arrayImg = (ArrayImg<IntType, ?>) image;
		final Object dataAccess = arrayImg.update(null);
		return dataAccess instanceof IntArray ? //
			((IntArray) dataAccess).getCurrentStorageArray() : null;
	}

	private static long[] extractLongArray(
		final RandomAccessibleInterval<LongType> image)
	{
		if (!(image instanceof ArrayImg)) return null;
		@SuppressWarnings("unchecked")
		final ArrayImg<LongType, ?> arrayImg = (ArrayImg<LongType, ?>) image;
		final Object dataAccess = arrayImg.update(null);
		return dataAccess instanceof LongArray ? //
			((LongArray) dataAccess).getCurrentStorageArray() : null;
	}

	private static <T extends RealType<T>> void copy(
		final RandomAccessibleInterval<T> source,
		final IterableInterval<T> dest)
	{
		final RandomAccess<T> sourceAccess = source.randomAccess();
		final Cursor<T> destCursor = dest.localizingCursor();
		while (destCursor.hasNext()) {
			destCursor.fwd();
			sourceAccess.setPosition(destCursor);
			destCursor.get().set(sourceAccess.get());
		}
	}
}
