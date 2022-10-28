package org.bioimageanalysis.icy.deeplearning.numpy;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ByteArrayUtils {

	
	/**
	 * Converts byte array into a signed integer 16 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a integer 16 buffer containing the wanted data
	 */
	public static short[] convertIntoSignedShort16(byte[] arr) {
		short[] int16 = new short[arr.length / 2];
		for ( int i = 0; i < arr.length / 4; i ++) {
			byte[] intArr = new byte[2];
			intArr[0] = arr[i * 2];
			intArr[1] = arr[i * 4 + 1];
			int16[i] = ByteBuffer.wrap(intArr).order(ByteOrder.LITTLE_ENDIAN).getShort();
		}
		return int16;
	}
	
	/**
	 * Converts byte array into a signed integer 32 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a int array containing the wanted data
	 */
	public static int[] convertIntoSignedInt32(byte[] arr) {
		int[] int32 = new int[arr.length / 4];
		for ( int i = 0; i < arr.length / 4; i ++) {
			byte[] intArr = new byte[4];
			intArr[0] = arr[i * 4];
			intArr[1] = arr[i * 4 + 1];
			intArr[2] = arr[i * 4 + 2];
			intArr[3] = arr[i * 4 + 3];
			int32[i] = ByteBuffer.wrap(intArr).order(ByteOrder.LITTLE_ENDIAN).getInt();
		}
		return int32;
	}
	
	/**
	 * Converts byte array into a unsigned integer 16 bit array stored in 
	 * a buffer.
	 *  However, as this data type does not exist in Java, the values are stored
	 * in an int32 array containing the values that would correspond to
	 * an uin16 array
	 * @param arr
	 * 	the byte array
	 * @return an int array containing the wanted data
	 */
	public static int[] convertIntoUnignedIn16(byte[] arr) {
		short[] signedShortArr = convertIntoSignedShort16(arr);
		int[] uint16 = new int[signedShortArr.length];
		int nVals = (2^16);
		for (int i = 0; i < signedShortArr.length; i ++) {
			if (signedShortArr[i] < 0) {
				uint16[i] = nVals + signedShortArr[i];
				continue;
			}
			uint16[i] = signedShortArr[i];
		}
		for ( int i = 0; i < arr.length / 2; i ++) {
			byte[] shortArr = new byte[2];
			shortArr[0] = arr[i * 2];
			shortArr[1] = arr[i * 2 + 1];
			uint16[i] = (int)(
			        (int)(shortArr[1] & 0xff) << 8  |
			        (int)(shortArr[0] & 0xff)
			         );
		}
		return uint16;
	}
	
	/**
	 * Converts byte array into a signed float 32 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a float arr containing the wanted data
	 */
	public static float[] convertIntoSignedFloat32(byte[] arr) {
		float[] float32 = new float[arr.length / 4];
		for ( int i = 0; i < arr.length / 4; i ++) {
			byte[] floatArr = new byte[4];
			floatArr[0] = arr[i * 4];
			floatArr[1] = arr[i * 4 + 1];
			floatArr[2] = arr[i * 4 + 2];
			floatArr[3] = arr[i * 4 + 3];
			float32[i] = ByteBuffer.wrap(floatArr).order(ByteOrder.LITTLE_ENDIAN).getFloat();
		}
		return float32;
	}
	
	/**
	 * Converts byte array into a signed float 64 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a double arr containing the wanted data
	 */
	public static double[] convertIntoSignedFloat64(byte[] arr) {
		double[] dd = new double[arr.length / 8];
		for ( int i = 0; i < arr.length / 8; i ++) {
			byte[] doubleArr = new byte[8];
			doubleArr[0] = arr[i * 8];
			doubleArr[1] = arr[i * 8 + 1];
			doubleArr[2] = arr[i * 8 + 2];
			doubleArr[3] = arr[i * 8 + 3];
			doubleArr[4] = arr[i * 8 + 4];
			doubleArr[5] = arr[i * 8 + 5];
			doubleArr[6] = arr[i * 8 + 6];
			doubleArr[7] = arr[i * 8 + 7];
			dd[i] = ByteBuffer.wrap(doubleArr).order(ByteOrder.LITTLE_ENDIAN).getDouble();
		}
		return dd;
	}
	
	/**
	 * Converts byte array into a signed float 64 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a long arr containing the wanted data
	 */
	public static long[] convertIntoSignedInt64(byte[] arr) {
		long[] dd = new long[arr.length / 8];
		for ( int i = 0; i < arr.length / 8; i ++) {
			byte[] doubleArr = new byte[8];
			doubleArr[0] = arr[i * 8];
			doubleArr[1] = arr[i * 8 + 1];
			doubleArr[2] = arr[i * 8 + 2];
			doubleArr[3] = arr[i * 8 + 3];
			doubleArr[4] = arr[i * 8 + 4];
			doubleArr[5] = arr[i * 8 + 5];
			doubleArr[6] = arr[i * 8 + 6];
			doubleArr[7] = arr[i * 8 + 7];
			dd[i] = ByteBuffer.wrap(doubleArr).order(ByteOrder.LITTLE_ENDIAN).getLong();
		}
		return dd;
	}
}
