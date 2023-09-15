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
import java.nio.ByteOrder;

public class ByteArrayUtils {

	
	/**
	 * Converts byte array into a signed integer 16 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a integer 16 buffer containing the wanted data
	 */
	public static short[] toInt16(byte[] arr) {
		return toInt16(arr, ByteOrder.LITTLE_ENDIAN);
	}

		
	/**
	 * Converts byte array into a signed integer 16 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @param byteOrder
	 * 	the order of the bytes in the array, LittleEndian or BigEndian
	 * @return a integer 16 buffer containing the wanted data
	 */
	public static short[] toInt16(byte[] arr, ByteOrder byteOrder) {
		short[] int16 = new short[arr.length / 2];
		for ( int i = 0; i < arr.length / 4; i ++) {
			byte[] intArr = new byte[2];
			intArr[0] = arr[i * 2];
			intArr[1] = arr[i * 4 + 1];
			int16[i] = ByteBuffer.wrap(intArr).order(byteOrder).getShort();
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
	public static int[] toUInt8(byte[] arr) {
		return toUInt8(arr, ByteOrder.LITTLE_ENDIAN);
	}
			
	/**
	 * Converts byte array into a signed integer 32 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @param byteOrder
	 * 	the order of the bytes in the array, LittleEndian or BigEndian
	 * @return a int array containing the wanted data
	 */
	public static int[] toUInt8(byte[] arr, ByteOrder byteOrder) {
		int[] int32 = new int[arr.length];
		for ( int i = 0; i < arr.length; i ++) {
			if (arr[i] < 0)
				int32[i] = 256 + arr[i];
			else
				int32[i] = arr[i];
		}
		return int32;
	}
	
	/**
	* Converts byte array into a signed integer 32 bit array stored in 
	* a buffer.
	* @param arr
	* 	the byte array
	* @return a int array containing the wanted data
	*/
	public static int[] toInt32(byte[] arr) {
		return toInt32(arr, ByteOrder.LITTLE_ENDIAN);
	}
		
	/**
	* Converts byte array into a signed integer 32 bit array stored in 
	* a buffer.
	* @param arr
	* 	the byte array
	 * @param byteOrder
	 * 	the order of the bytes in the array, LittleEndian or BigEndian
	* @return a int array containing the wanted data
	*/
	public static int[] toInt32(byte[] arr, ByteOrder byteOrder) {
		int[] int32 = new int[arr.length / 4];
		for ( int i = 0; i < arr.length / 4; i ++) {
			byte[] intArr = new byte[4];
			intArr[0] = arr[i * 4];
			intArr[1] = arr[i * 4 + 1];
			intArr[2] = arr[i * 4 + 2];
			intArr[3] = arr[i * 4 + 3];
			int32[i] = ByteBuffer.wrap(intArr).order(byteOrder).getInt();
		}
		return int32;
	}
	
	/**
	 * Converts byte array into a unsigned integer 32 bit array stored in 
	 * a buffer.
	 *  However, as this data type does not exist in Java, the values are stored
	 * in an long (int64) array containing the values that would correspond to
	 * an uin32 array
	 * @param arr
	 * 	the byte array
	 * @return an long array containing the wanted data
	 */
	public static long[] toUInt32(byte[] arr) {
		return toUInt32(arr, ByteOrder.LITTLE_ENDIAN);
	}
	
	/**
	 * Converts byte array into a unsigned integer 32 bit array stored in 
	 * a buffer.
	 *  However, as this data type does not exist in Java, the values are stored
	 * in an long (int64) array containing the values that would correspond to
	 * an uin32 array
	 * @param arr
	 * 	the byte array
	 * @param byteOrder
	 * 	the order of the bytes in the array, LittleEndian or BigEndian
	 * @return an long array containing the wanted data
	 */
	public static long[] toUInt32(byte[] arr, ByteOrder byteOrder) {
		long[] uint32 = new long[arr.length / 4];
		for ( int i = 0; i < arr.length / 4; i ++) {
			byte[] intArr = new byte[4];
			intArr[0] = arr[i * 4];
			intArr[1] = arr[i * 4 + 1];
			intArr[2] = arr[i * 4 + 2];
			intArr[3] = arr[i * 4 + 3];
			int number = ByteBuffer.wrap(intArr).order(byteOrder).getInt();
			if (number < 0)
				uint32[i] = (long) (Math.pow(2, 32) + number);
			else
				uint32[i] = number;
		}
		return uint32;
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
	public static int[] toUInt16(byte[] arr) {
		return toUInt16(arr, ByteOrder.LITTLE_ENDIAN);
	}
	
	/**
	 * Converts byte array into a unsigned integer 16 bit array stored in 
	 * a buffer.
	 *  However, as this data type does not exist in Java, the values are stored
	 * in an int32 array containing the values that would correspond to
	 * an uin16 array
	 * @param arr
	 * 	the byte array
	 * @param byteOrder
	 * 	the order of the bytes in the array, LittleEndian or BigEndian
	 * @return an int array containing the wanted data
	 */
	public static int[] toUInt16(byte[] arr, ByteOrder byteOrder) {
		int[] int16 = new int[arr.length / 2];
		for ( int i = 0; i < arr.length / 4; i ++) {
			byte[] intArr = new byte[2];
			intArr[0] = arr[i * 2];
			intArr[1] = arr[i * 4 + 1];
			short number = ByteBuffer.wrap(intArr).order(byteOrder).getShort();
			if (number < 0)
				int16[i] = (int) (Math.pow(2, 16) + number);
			else
				int16[i] = number;
		}
		return int16;
	}
	
	/**
	 * Converts byte array into a signed float 32 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a float arr containing the wanted data
	 */
	public static float[] toFloat32(byte[] arr) {
		return toFloat32(arr, ByteOrder.LITTLE_ENDIAN);
	}
	
	/**
	 * Converts byte array into a signed float 32 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @param byteOrder
	 * 	the order of the bytes in the array, LittleEndian or BigEndian
	 * @return a float arr containing the wanted data
	 */
	public static float[] toFloat32(byte[] arr, ByteOrder byteOrder) {
		float[] float32 = new float[arr.length / 4];
		for ( int i = 0; i < arr.length / 4; i ++) {
			byte[] floatArr = new byte[4];
			floatArr[0] = arr[i * 4];
			floatArr[1] = arr[i * 4 + 1];
			floatArr[2] = arr[i * 4 + 2];
			floatArr[3] = arr[i * 4 + 3];
			float32[i] = ByteBuffer.wrap(floatArr).order(byteOrder).getFloat();
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
	public static double[] toFloat64(byte[] arr) {
		return toFloat64(arr, ByteOrder.LITTLE_ENDIAN);
	}
	
	/**
	 * Converts byte array into a signed float 64 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @param byteOrder
	 * 	the order of the bytes in the array, LittleEndian or BigEndian
	 * @return a double arr containing the wanted data
	 */
	public static double[] toFloat64(byte[] arr, ByteOrder byteOrder) {
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
			dd[i] = ByteBuffer.wrap(doubleArr).order(byteOrder).getDouble();
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
	public static long[] toInt64(byte[] arr) {
		return toInt64(arr, ByteOrder.LITTLE_ENDIAN);
	}
	
	/**
	 * Converts byte array into a signed float 64 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @param byteOrder
	 * 	the order of the bytes in the array, LittleEndian or BigEndian
	 * @return a long arr containing the wanted data
	 */
	public static long[] toInt64(byte[] arr, ByteOrder byteOrder) {
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
			dd[i] = ByteBuffer.wrap(doubleArr).order(byteOrder).getLong();
		}
		return dd;
	}
	
	/**
	 * Converts byte array into a boolean array
	 * @param arr
	 * 	the byte array
	 * @param byteOrder
	 * 	the order of the bytes in the array, LittleEndian or BigEndian
	 * @return a long boolean containing the wanted data
	 */
	public static boolean[] toBoolean(byte[] arr, ByteOrder byteOrder) {
		return toBoolean(arr);
	}
	
	/**
	 * Converts byte array into a boolean array
	 * @param arr
	 * 	the byte array
	 * @return a long boolean containing the wanted data
	 */
	public static boolean[] toBoolean(byte[] arr) {
		boolean[] dd = new boolean[arr.length];
		for ( int i = 0; i < arr.length; i ++) {
			dd[i] = arr[i] != 0;
		}
		return dd;
	}
}
