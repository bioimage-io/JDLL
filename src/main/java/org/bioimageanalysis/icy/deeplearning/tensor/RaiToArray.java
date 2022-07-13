package org.bioimageanalysis.icy.deeplearning.tensor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

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
	
	public float[] convertIntArrIntoFloatArr(int[] intArr) {
		float[] arr = new float[intArr.length];
		for (int i = 0; i < intArr.length; i ++)
			arr[i] = (float) intArr[i];
		return arr;
	}
	
	public float[] convertDoubleArrIntoFloatArr(double[] doubleArr) {
		float[] arr = new float[doubleArr.length];
		for (int i = 0; i < doubleArr.length; i ++)
			arr[i] = (float) doubleArr[i];
		return arr;
	}
	
	public float[] convertByteArrIntoFloatArr(byte[] byteArr) {
		float[] float32 = new float[byteArr.length / 4];
		for ( int i = 0; i < byteArr.length / 4; i ++) {
			byte[] floatArr = new byte[4];
			floatArr[0] = byteArr[i * 4];
			floatArr[1] = byteArr[i * 4 + 1];
			floatArr[2] = byteArr[i * 4 + 2];
			floatArr[3] = byteArr[i * 4 + 3];
			float32[i] = ByteBuffer.wrap(floatArr).order(ByteOrder.LITTLE_ENDIAN).getFloat();
		}
		return float32;
	}
	
	public float[] convertLongArrIntoFloatArr(long[] longArr) {
		float[] arr = new float[longArr.length];
		for (int i = 0; i < longArr.length; i ++)
			arr[i] = (float) longArr[i];
		return arr;
	}
	
	public int[] convertFloatArrIntoIntArr(float[] floatArr) {
		int[] arr = new int[floatArr.length];
		for (int i = 0; i < floatArr.length; i ++)
			arr[i] = (int) floatArr[i];
		return arr;
	}
	
	public int[] convertDoubleArrIntoIntArr(double[] doubleArr) {
		int[] arr = new int[doubleArr.length];
		for (int i = 0; i < doubleArr.length; i ++)
			arr[i] = (int) doubleArr[i];
		return arr;
	}
	
	public int[] convertByteArrIntoIntArr(byte[] byteArr) {
		int[] int32 = new int[byteArr.length / 4];
		for ( int i = 0; i < byteArr.length / 4; i ++) {
			byte[] intArr = new byte[4];
			intArr[0] = byteArr[i * 4];
			intArr[1] = byteArr[i * 4 + 1];
			intArr[2] = byteArr[i * 4 + 2];
			intArr[3] = byteArr[i * 4 + 3];
			int32[i] = ByteBuffer.wrap(intArr).order(ByteOrder.LITTLE_ENDIAN).getInt();
		}
		return int32;
	}
	
	public int[] convertLongArrIntoIntArr(long[] longArr) {
		int[] arr = new int[longArr.length];
		for (int i = 0; i < longArr.length; i ++)
			arr[i] = (int) longArr[i];
		return arr;
	}
	
	public double[] convertIntArrIntoDoubleArr(int[] intArr) {
		double[] arr = new double[intArr.length];
		for (int i = 0; i < intArr.length; i ++)
			arr[i] = (double) intArr[i];
		return arr;
	}
	
	public double[] convertFloatArrIntoDoubleArr(float[] floatArr) {
		double[] arr = new double[floatArr.length];
		for (int i = 0; i < floatArr.length; i ++)
			arr[i] = (double) floatArr[i];
		return arr;
	}
	
	public double[] convertByteArrIntoDoubleArr(byte[] byteArr) {
		double[] dd = new double[byteArr.length / 8];
		for ( int i = 0; i < byteArr.length / 8; i ++) {
			byte[] doubleArr = new byte[8];
			doubleArr[0] = byteArr[i * 8];
			doubleArr[1] = byteArr[i * 8 + 1];
			doubleArr[2] = byteArr[i * 8 + 2];
			doubleArr[3] = byteArr[i * 8 + 3];
			doubleArr[4] = byteArr[i * 8 + 4];
			doubleArr[5] = byteArr[i * 8 + 5];
			doubleArr[6] = byteArr[i * 8 + 6];
			doubleArr[7] = byteArr[i * 8 + 7];
			dd[i] = ByteBuffer.wrap(doubleArr).order(ByteOrder.LITTLE_ENDIAN).getDouble();
		}
		return dd;
	}
	
	public double[] convertLongArrIntoDoubleArr(long[] longArr) {
		double[] arr = new double[longArr.length];
		for (int i = 0; i < longArr.length; i ++)
			arr[i] = (double) longArr
			[i];
		return arr;
	}
	
	public byte[] convertIntArrIntoByteArr(int[] intArr) {

		return null;
	}
	
	public byte[] convertDoubleArrIntoByteArr(double[] doubleArr) {

		return null;
	}
	
	public byte[] convertFloatArrIntoByteArr(float[] byteArr) {

		return null;
	}
	
	public byte[] convertLongArrIntoByteArr(long[] longArr) {

		return null;
	}
	
	public long[] convertIntArrIntoLongArr(int[] intArr) {

		return null;
	}
	
	public long[] convertDoubleArrIntoLongArr(double[] doubleArr) {

		return null;
	}
	
	public long[] convertByteArrIntoLongArr(byte[] byteArr) {
		long[] dd = new long[byteArr.length / 8];
		for ( int i = 0; i < byteArr.length / 8; i ++) {
			byte[] doubleArr = new byte[8];
			doubleArr[0] = byteArr[i * 8];
			doubleArr[1] = byteArr[i * 8 + 1];
			doubleArr[2] = byteArr[i * 8 + 2];
			doubleArr[3] = byteArr[i * 8 + 3];
			doubleArr[4] = byteArr[i * 8 + 4];
			doubleArr[5] = byteArr[i * 8 + 5];
			doubleArr[6] = byteArr[i * 8 + 6];
			doubleArr[7] = byteArr[i * 8 + 7];
			dd[i] = ByteBuffer.wrap(doubleArr).order(ByteOrder.LITTLE_ENDIAN).getLong();
		}
		return dd;
	}
	
	public long[] convertLongArrIntoLongArr(float[] longArr) {
		return null;
	}
	
}
