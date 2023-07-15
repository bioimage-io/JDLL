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
package io.bioimage.modelrunner.bioimageio.bioengine.tensor;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.HashMap;

import io.bioimage.modelrunner.numpy.ByteArrayUtils;

/**
 * Class that converts each of the particular output arrays produced by the BioEngine
 * server into readable buffers with their corresponding shape and data type.
 * The produced object can be used to create an image easily.
 * 
 * @author Carlos Javier García López de Haro
 */
public class BioEngineOutputArray {
	/**
	 * Name of the output image array
	 */
	private String name;
	/**
	 * Array containing the shape of the output array
	 */
	private int[] shape;
	/**
	 * String containing the data type of the array.
	 * Look at TODO to see data types supported
	 */
	private String dtype;
	/**
	 * Key for the boolean data type
	 */
	private static String boolKey = "bool";
	/**
	 * Buffer containing the data of the output array of the BioEngine
	 */
	private Buffer dataBuffer;
	
	/** TODO add possibility of having parameters
	 * TODO we need the shape of the array too
	 * Manage ouputs from the BioEngine to be able to be handled by a common
	 * Java consumer
	 * @param name
	 * 	Name of the output
	 * @param outputMap
	 * 	Map containing the data produced by the BioEngine for a particular output
	 * @throws IllegalArgumentException if the data type of the array is not supported
	 */
	private BioEngineOutputArray(String name, HashMap<String, Object> outputMap) throws IllegalArgumentException {
		setName(name);
		setShape(outputMap.get(BioengineTensor.SHAPE_KEY));
		setDType((String) outputMap.get(BioengineTensor.DTYPE_KEY));
		setArray(outputMap.get(BioengineTensor.VALUE_KEY));
	}
	
	/**
	 * Creates a BioEngine output array in a readable way for Java consumers
	 * @param name
	 * 	name of the output
	 * @param dataType
	 * 	data type of the tensor
	 * @param shape
	 * 	shape of the tensor
	 * @param buffer
	 * 	data of the tensor as a byte array
	 * @return an understandable tensor
	 * @throws IllegalArgumentException if the data type of the array is not supported
	 */
	public BioEngineOutputArray(String name, String dataType, Object shape, byte[] buffer) {
		setName(name);
		setShape(shape);
		setDType(dataType);
		setArray(buffer);
	}

	/**
	 * Create an array from the bytes received by the BioEngine and using the corresponding shape
	 * and data types
	 * @param byteArray
	 * 	the  byte array that contains the array data
	 * @throws IllegalArgumentException if the data type of the array is not supported
	 */
	private void setArray(Object byteArrayObject) throws IllegalArgumentException {
		byte[] byteArray = null;;
		try {
			byteArray = getByteArray(byteArrayObject);
		} catch (Exception ex) {
			throw new IllegalArgumentException("Error retrieving information from the BioEngine output '" + this.name + "'.\n"
					+ "The array data is not correctly defined and cannot be read, it should be either a byte array or List of bytes.");
		}
		
		if (this.dtype.toLowerCase().equals(BioengineTensor.FLOAT64_STR)) {
			this.dataBuffer = convertIntoSignedFloat64(byteArray);
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.FLOAT32_STR)) {
			this.dataBuffer = convertIntoSignedFloat32(byteArray);
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.INT32_STR)) {
			this.dataBuffer = convertIntoSignedInt32(byteArray);
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.INT16_STR)) {
			this.dataBuffer = convertIntoSignedInt16(byteArray);
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.UINT16_STR)) {
			this.dataBuffer = convertIntoUnsignedInt16(byteArray);
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.BYTE_STR)) {
			this.dataBuffer = convertIntoSignedInt8(byteArray);
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.UBYTE_STR)) {
			this.dataBuffer = convertIntoUnignedInt8(byteArray);
		} else if (this.dtype.toLowerCase().equals(boolKey)) {
			this.dataBuffer = convertIntoBoolean(byteArray);
		} else {
			throw new IllegalArgumentException("Output array '" + this.name +"' could not be retrieved.\n"
					+ "Its corresponding data type '" + this.dtype + "' is not supported yet.");
		}
	}
	
	/**
	 * Retrieve the buffer containing the data of the output array produced
	 * by the BioEngine
	 * @return the buffer with the output array
	 */
	public Buffer getArray() {
		return this.dataBuffer;
	}
	
	/**
	 * Converts byte array into a signed integer 16 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a integer 16 buffer containing the wanted data
	 */
	public static ShortBuffer convertIntoSignedInt16(byte[] arr) {
		return ShortBuffer.wrap(ByteArrayUtils.convertIntoSignedShort16(arr));
	}
	
	/**
	 * Converts byte array into a signed integer 32 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a int buffer containing the wanted data
	 */
	public static IntBuffer convertIntoSignedInt32(byte[] arr) {
		return IntBuffer.wrap(ByteArrayUtils.convertIntoSignedInt32(arr));
	}
	
	/**
	 * Converts byte array into a boolean array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a int buffer containing the wanted boolean data
	 */
	public static IntBuffer convertIntoBoolean(byte[] arr) {
		int[] boolArr = new int[arr.length];
		for (int i = 0; i < arr.length; i ++) {
			boolArr[i] = (int) arr[i];
		}
		return IntBuffer.wrap(boolArr);
	}
	
	/**
	 * Converts byte array into a unsigned integer 16 bit array stored in 
	 * a buffer.
	 *  However, as this data type does not exist in Java, the values are stored
	 * in an int32 array containing the values that would correspond to
	 * an uin16 array
	 * @param arr
	 * 	the byte array
	 * @return an int buffer containing the wanted data
	 */
	public static IntBuffer convertIntoUnsignedInt16(byte[] arr) {
		return IntBuffer.wrap(ByteArrayUtils.convertIntoUnsignedInt16(arr));
	}
	
	/**
	 * Converts byte array into a signed integer 8 bit array stored in 
	 * a buffer. 
	 * However, as this data type does not exist in Java, the values are stored
	 * in an int32 array containing the values that would correspond to
	 * an int8 array
	 * @param arr
	 * @param arr
	 * 	the byte array
	 * @return an int buffer containing the wanted data
	 */
	public static ByteBuffer convertIntoSignedInt8(byte[] arr) {
		return ByteBuffer.wrap(arr);
	}
	
	/**
	 * Converts byte array into an unsigned integer 8 bit array stored in 
	 * a buffer. However, as this data type does not exist in Java, the values are stored
	 * in an int32 array containing the values that would correspond to
	 * an uint8 array
	 * @param arr
	 * @param arr
	 * 	the byte array
	 * @return an int buffer containing the wanted data
	 */
	public static IntBuffer convertIntoUnignedInt8(byte[] arr) {
		int[] uint8 = new int[arr.length];
		for (int i = 0; i < arr.length; i ++) {
			uint8[i] = arr[i] & 0xff;
		}
		return IntBuffer.wrap(uint8);
	}
	
	/**
	 * Converts byte array into a signed float 32 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a float buffer containing the wanted data
	 */
	public static FloatBuffer convertIntoSignedFloat32(byte[] arr) {
		return FloatBuffer.wrap(ByteArrayUtils.convertIntoSignedFloat32(arr));
	}
	
	/**
	 * Converts byte array into a signed float 64 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a double buffer containing the wanted data
	 */
	public static DoubleBuffer convertIntoSignedFloat64(byte[] arr) {
		return DoubleBuffer.wrap(ByteArrayUtils.convertIntoSignedFloat64(arr));
	}
	
	/**
	 * Creates a BioEngine output array in a readable way for Java consumers
	 * @param name
	 * 	name of the output
	 * @param outputMap
	 * 	Map containing the information about the output
	 * @return an object with ordered information about the output that can be managed
	 * 	by the consumer software
	 * @throws IllegalArgumentException if the data type of the array is not supported
	 */
	public static BioEngineOutputArray buildOutput(String name, HashMap<String, Object> outputMap) throws IllegalArgumentException {
		return new BioEngineOutputArray(name, outputMap);
	}
	
	/**
	 * Creates a BioEngine output array in a readable way for Java consumers
	 * @param name
	 * 	name of the output
	 * @param dataType
	 * 	data type of the tensor
	 * @param shape
	 * 	shape of the tensor
	 * @param buffer
	 * 	data of the tensor as a byte array
	 * @return an understandable tensor
	 * @throws IllegalArgumentException if the data type of the array is not supported
	 */
	public static BioEngineOutputArray buildOutput(String name, String dataType, Object shape, byte[] buffer) throws IllegalArgumentException {
		return new BioEngineOutputArray(name, dataType, shape, buffer);
	}
	
	/**
	 * Sets the data type of the array
	 * @param dtype
	 * 	 the data type
	 */
	private void setDType(String dtype) {
		this.dtype = dtype;
	}
	
	/**
	 * Gets the data type of the array
	 * @return the data type of the array
	 */
	public String getDType() {
		return this.dtype;
	}
	
	/**
	 * Sets the shape of the array
	 * @param shape
	 * 	the shape of the array
	 * @throws IllegalArgumentException if the shape object does not correspond to an array
	 */
	private void setShape(Object shape) throws IllegalArgumentException{
		try {
			this.shape = getIntArray(shape);
		} catch (Exception ex) {
			throw new IllegalArgumentException("Error retrieving information from the BioEngine output '" + this.name + "'.\n"
					+ "The shape is not correctly defined, it should be either an int array or ArrayList<Integer>.");
		}
	}
	
	/**
	 * Gets the shape of the array
	 * @return the shape of the array
	 */
	public int[] getShape() {
		return this.shape;
	}
	
	/**
	 * Sets the name of the array
	 * @param name
	 * 	the name of the array
	 */
	private void setName(String name) {
		this.name = name;
	}
	
	/**
	 * Gets the name of the array
	 * @return the name of the array
	 */
	public String getName() {
		return this.name;
	}
	
	/**
	 * Casts an int array in an object that contains an int array
	 * @param shape
	 * 	the object containing teh int array
	 * @return teh int array
	 * @throws Exception if it was impossible to cast the array from the object
	 */
	public int[] getIntArray(Object shape) throws Exception {
		int[] shapeArr = null;
		if (shape != null && shape instanceof ArrayList<?>) {
			ArrayList<?> ll = (ArrayList<?>) shape;
			shapeArr = new int [ll.size()];
			for (int c = 0; c < ll.size(); c ++)
				shapeArr[c] = castUnknownToInt(ll.get(c));			
		} else if (shape != null&& shape instanceof int[]){
			shapeArr = (int[]) shape;
		} else if (shape != null&& shape instanceof double[]){
			double[] shapeArrDouble = (double[]) shape;
			shapeArr = new int[shapeArrDouble.length];
			for (int i = 0; i < shapeArrDouble.length; i ++) {
				shapeArr[i] = (int) shapeArrDouble[i];
			}
		} else {
			throw new Exception("Datatype of shape array cannot be casted to int or double.");
		}
		return shapeArr;
	}
	
	/**
	 * Cast unknown number to int
	 * @param unknownNumber
	 * 	the unknown number
	 * @return the int
	 */
	private int castUnknownToInt(Object unknownNumber) {
		if (unknownNumber instanceof Integer){
			return (int) unknownNumber;
		} else if (unknownNumber instanceof Double){
			return ((Double) unknownNumber).intValue();
		} else if (unknownNumber instanceof Float){
			return ((Float) unknownNumber).intValue();
		} else if (unknownNumber instanceof Short){
			return ((Short) unknownNumber).intValue();
		} else if (unknownNumber instanceof Byte){
			return ((Byte) unknownNumber).intValue();
		} else if (unknownNumber instanceof Long){
			return ((Long) unknownNumber).intValue();
		} else if (unknownNumber instanceof Number){
			return ((Number) unknownNumber).intValue();
		} else {
			throw new IllegalArgumentException("Shape of the output '" + this.name
								+ "' is not defined with allowed Java types.");
		}
	}
	
	/**
	 * Casts a byte array in an object that contains a byte array
	 * @param listArr
	 * 	the object containing the byte array
	 * @return the byte array
	 * @throws Exception if it was impossible to cast the array from the object
	 */
	public static byte[] getByteArray(Object listArr) throws Exception {
		byte[] arr = null;
		if (listArr != null && listArr instanceof ArrayList<?>) {
			@SuppressWarnings("unchecked")
			ArrayList<Byte> ll = (ArrayList<Byte>) listArr;
			arr = new byte[ll.size()];
			for (byte c = 0; c < ll.size(); c ++)
				arr[c] = (byte) ll.get(c);			
		} else if (listArr != null && listArr instanceof byte[]){
			arr = (byte[]) listArr;
		} else {
			throw new Exception();
		}
		return arr;
	}
}
