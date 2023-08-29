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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Objects;

import io.bioimage.modelrunner.numpy.ByteArrayUtils;
import io.bioimage.modelrunner.utils.IndexingUtils;
import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.ByteAccess;
import net.imglib2.img.basictypeaccess.DoubleAccess;
import net.imglib2.img.basictypeaccess.FloatAccess;
import net.imglib2.img.basictypeaccess.IntAccess;
import net.imglib2.img.basictypeaccess.LongAccess;
import net.imglib2.img.basictypeaccess.ShortAccess;
import net.imglib2.img.basictypeaccess.nio.ByteBufferAccess;
import net.imglib2.img.basictypeaccess.nio.DoubleBufferAccess;
import net.imglib2.img.basictypeaccess.nio.FloatBufferAccess;
import net.imglib2.img.basictypeaccess.nio.IntBufferAccess;
import net.imglib2.img.basictypeaccess.nio.LongBufferAccess;
import net.imglib2.img.basictypeaccess.nio.ShortBufferAccess;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

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
	private long[] shape;
	/**
	 * String containing the data type of the array.
	 * Look at TODO to see data types supported
	 */
	private String dtype;
	/**
	 * byte array containing the data of the array
	 */
	private byte[] arr;
	
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
		setData(outputMap.get(BioengineTensor.VALUE_KEY));
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
		setData(buffer);
	}

	/**
	 * REtrieve the byte array that contains the data of the tensor
	 * @param byteArray
	 * 	the  byte array that contains the array data
	 * @throws IllegalArgumentException if the data type of the array is not supported
	 */
	private void setData(Object byteArrayObject) throws IllegalArgumentException {
		try {
			arr = getByteArray(byteArrayObject);
		} catch (Exception ex) {
			throw new IllegalArgumentException("Error retrieving information from the BioEngine output '" + this.name + "'.\n"
					+ "The array data is not correctly defined and cannot be read, it should be either a byte array or List of bytes.");
		}
	}


	/**
	 * Create an array from the bytes received by the BioEngine and using the corresponding shape
	 * and data types
	 * @return an ImgLib2 {@link Img} containing the data of one of the outputs of the bioengine
	 * @throws IllegalArgumentException if the data type of the array is not supported
	 */
	@SuppressWarnings("unchecked")
	public < T extends RealType< T > & NativeType< T > >  Img<T> getImg()
			throws IllegalArgumentException {
		return getImg(ByteOrder.LITTLE_ENDIAN);
	}


	/**
	 * Create an array from the bytes received by the BioEngine and using the corresponding shape
	 * and data types
	 * @param byteOrder
	 * 	order of the bytes
	 * @return an ImgLib2 {@link Img} containing the data of one of the outputs of the bioengine
	 * @throws IllegalArgumentException if the data type of the array is not supported
	 */
	@SuppressWarnings("unchecked")
	public < T extends RealType< T > & NativeType< T > >  Img<T> getImg(ByteOrder byteOrder)
			throws IllegalArgumentException {
		Objects.requireNonNull(arr);
		ByteBuffer buf = ByteBuffer.wrap(arr).order(byteOrder);
		if (this.dtype.toLowerCase().equals(BioengineTensor.FLOAT64_STR)) {
    		DoubleAccess access = new DoubleBufferAccess(buf, true);
    		return (Img<T>) ArrayImgs.doubles( access, shape );
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.INT64_STR)) {
    		LongAccess access = new LongBufferAccess(buf, true);
    		return (Img<T>) ArrayImgs.longs( access, shape );
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.FLOAT32_STR)) {
			
			
			final ArrayImgFactory< FloatType > factory = new ArrayImgFactory<>( new FloatType() );
	        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(shape);
	    	Cursor<FloatType> tensorCursor= outputImg.cursor();
	    	long flatSize = 1;
	    	for (long l : shape) {flatSize *= l;}
	    	float[] flatArr = ByteArrayUtils.toFloat32(arr);
			while (tensorCursor.hasNext()) {
				tensorCursor.fwd();
				long[] cursorPos = tensorCursor.positionAsLongArray();
	        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, shape);
	        	float val = flatArr[flatPos];
	        	tensorCursor.get().set(val);
			}
		 	return (Img<T>) outputImg;
			
			
			
			
			
    		//FloatAccess access = new FloatBufferAccess(buf, true);
    		//return (Img<T>) ArrayImgs.floats( access, shape );
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.INT32_STR)) {
    		IntAccess access = new IntBufferAccess(buf, true);
    		return (Img<T>) ArrayImgs.ints( access, shape );
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.UINT32_STR)) {
    		IntAccess access = new IntBufferAccess(buf, true);
    		return (Img<T>) ArrayImgs.unsignedInts( access, shape );
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.INT16_STR)) {
    		ShortAccess access = new ShortBufferAccess(buf, true);
    		return (Img<T>) ArrayImgs.shorts( access, shape );
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.UINT16_STR)) {
    		ShortAccess access = new ShortBufferAccess(buf, true);
    		return (Img<T>) ArrayImgs.unsignedShorts( access, shape );
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.BYTE_STR)) {
    		ByteAccess access = new ByteBufferAccess(buf, true);
    		return (Img<T>) ArrayImgs.bytes( access, shape );
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.UBYTE_STR)) {
    		ByteAccess access = new ByteBufferAccess(buf, true);
    		return (Img<T>) ArrayImgs.unsignedBytes( access, shape );
		} else if (this.dtype.toLowerCase().equals(BioengineTensor.BOOL_STR)) {
    		return (Img<T>) ArrayImgs.booleans(ByteArrayUtils.toBoolean(arr), shape );
		} else {
			throw new IllegalArgumentException("Output array '" + this.name +"' could not be retrieved.\n"
					+ "Its corresponding data type '" + this.dtype + "' is not supported yet.");
		}
	}
	
	/**
	 * Retrieve the byte array containing the data of the tensor
	 * @return the byte array with the data of the tensor
	 */
	public byte[] getArray() {
		return this.arr;
	}
	
	/**
	 * Converts byte array into a signed integer 16 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a integer 16 buffer containing the wanted data
	 */
	public static ShortBuffer convertIntoSignedInt16(byte[] arr) {
		return ShortBuffer.wrap(ByteArrayUtils.toInt16(arr));
	}
	
	/**
	 * Converts byte array into a signed integer 32 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a int buffer containing the wanted data
	 */
	public static IntBuffer convertIntoSignedInt32(byte[] arr) {
		return IntBuffer.wrap(ByteArrayUtils.toInt32(arr));
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
		return IntBuffer.wrap(ByteArrayUtils.toUInt16(arr));
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
		return FloatBuffer.wrap(ByteArrayUtils.toFloat32(arr));
	}
	
	/**
	 * Converts byte array into a signed float 64 bit array stored in 
	 * a buffer.
	 * @param arr
	 * 	the byte array
	 * @return a double buffer containing the wanted data
	 */
	public static DoubleBuffer convertIntoSignedFloat64(byte[] arr) {
		return DoubleBuffer.wrap(ByteArrayUtils.toFloat64(arr));
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
			this.shape = getLongArray(shape);
		} catch (Exception ex) {
			throw new IllegalArgumentException("Error retrieving information from the BioEngine output '" + this.name + "'.\n"
					+ "The shape is not correctly defined, it should be either an int array or ArrayList<Integer>.");
		}
	}
	
	/**
	 * Gets the shape of the array
	 * @return the shape of the array
	 */
	public long[] getShape() {
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
	 * Casts an long array in an object that contains an long array
	 * @param shape
	 * 	the object containing the long array
	 * @return the long array
	 * @throws Exception if it was impossible to cast the array from the object
	 */
	public long[] getLongArray(Object shape) throws Exception {
		long[] shapeArr = null;
		if (shape != null && shape instanceof ArrayList<?>) {
			ArrayList<?> ll = (ArrayList<?>) shape;
			shapeArr = new long [ll.size()];
			for (int c = 0; c < ll.size(); c ++)
				shapeArr[c] = castUnknownToLong(ll.get(c));			
		} else if (shape != null&& shape instanceof int[]){
			shapeArr = (long[]) shape;
		} else if (shape != null&& shape instanceof double[]){
			double[] shapeArrDouble = (double[]) shape;
			shapeArr = new long[shapeArrDouble.length];
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
	 * Cast unknown number to long
	 * @param unknownNumber
	 * 	the unknown number
	 * @return the long
	 */
	private long castUnknownToLong(Object unknownNumber) {
		if (unknownNumber instanceof Integer){
			return (int) unknownNumber;
		} else if (unknownNumber instanceof Double){
			return ((Double) unknownNumber).longValue();
		} else if (unknownNumber instanceof Float){
			return ((Float) unknownNumber).longValue();
		} else if (unknownNumber instanceof Short){
			return ((Short) unknownNumber).longValue();
		} else if (unknownNumber instanceof Byte){
			return ((Byte) unknownNumber).longValue();
		} else if (unknownNumber instanceof Long){
			return ((Long) unknownNumber).longValue();
		} else if (unknownNumber instanceof Number){
			return ((Number) unknownNumber).longValue();
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
