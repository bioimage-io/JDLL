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
import java.util.HashMap;
import java.util.Map;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;

public class TensorBuilder {
	
	/**
	 * Map containing the instances needed to provide an input to the 
	 * server.
	 * The input needs to have:
	 *  -An entry called "inputs", whose value is another Map that contains
	 *   the info about the input tensors
	 *  -An entry called model_name with the name of the model
	 *  -A fixed entry called decoe_json that equals to true
	 */
	private Map<String, Object> inputs = new HashMap<String, Object>();
	/**
	 * String key corresponding to the type of object being specified
	 */
	private final static String OBJECT_KEY = "_rtype";
	/**
	 * Value corresponding to the type of the array in the
	 * {@link #inputs} map
	 */
	private static final String NDARRAY_VALUE = "ndarray";
	/**
	 * Value corresponding to a parameter
	 */
	private static final String PARAMETER_VALUE = "parameter";
	/**
	 * String key corresponding to the value of the array in the
	 * {@link #inputs} map
	 */
	private static final String VALUE_KEY = "_rvalue";
	/**
	 * String key corresponding to the shape of the array in the
	 * {@link #inputs} map
	 */
	private static final String SHAPE_KEY = "_rshape";
	/**
	 * String key corresponding to the dtype of the array in the
	 * {@link #inputs} map
	 */
	private static final String DTYPE_KEY = "_rdtype";
	/**
	 * String corresponding to the dtype of the array in the
	 * {@link #inputs} map
	 */
	private String dtypeVal;
	/**
	 * String used as tag for the float32 np dtype
	 */
	private static final String FLOAT32_STR = "float32";
	/**
	 * String used as tag for the float64 np dtype
	 */
	private static final String FLOAT64_STR = "float64";
	/**
	 * String used as tag for the byte or int8 np dtype
	 */
	private static final String BYTE_STR = "int8";
	/**
	 * String used as tag for the byte or int16 np dtype
	 */
	private static final String INT16_STR = "int16";
	/**
	 * String used as tag for the int32 np dtype
	 */
	private static final String INT32_STR = "int32";
	/**
	 * String used as tag for the int64 np dtype
	 */
	private static final String INT64_STR = "int64";
	/**
	 * String used as tag for the ubyte or uint8 np dtype
	 */
	private static final String UBYTE_STR = "uint8";
	/**
	 * String used as tag for the uint16 np dtype
	 */
	private static final String UINT16_STR = "uint16";
	/**
	 * String used as tag for the uint32 np dtype
	 */
	private static final String UINT32_STR = "uint32";

	/**
	 * Utility class.
	 */
	private TensorBuilder() {}

	
	public static  < T extends RealType< T > & NativeType< T > >
				byte[] createByteArray(Tensor<T> tensor) {
		return imglib2ToByteArray(tensor.getData());
	}

	
	public static < T extends RealType< T > & NativeType< T > >
				byte[] imglib2ToByteArray(RandomAccessibleInterval<T> rai) {
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
    		return buildByte((RandomAccessibleInterval<ByteType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedByteType) {
    		return buildUByte((RandomAccessibleInterval<UnsignedByteType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof ShortType) {
    		return buildShort((RandomAccessibleInterval<ShortType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedShortType) {
    		return buildUShort((RandomAccessibleInterval<UnsignedShortType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
    		return buildInt((RandomAccessibleInterval<IntType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedIntType) {
    		return buildUInt((RandomAccessibleInterval<UnsignedIntType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
    		return buildFloat((RandomAccessibleInterval<FloatType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
    		return buildDouble((RandomAccessibleInterval<DoubleType>) rai);
    	} else {
            throw new IllegalArgumentException("The image has an unsupported type: " + Util.getTypeFromInterval(rai).getClass().toString());
    	}
	}

    /**
     * Creates a byte array from a {@link ByteType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
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
     * Creates a byte array from a {@link UnsignedByteType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     */
    private static byte[] buildUByte(RandomAccessibleInterval<UnsignedByteType> imgTensor)
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
		byte[] byteArr = new byte[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int val = tensorCursor.get().get();
			if (val > 127)
				val = val - 256;
			byteArr[cc ++] = (byte) val;
		}
		return byteArr;
    }

    /**
     * Creates a byte array from a {@link ShortType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     */
    private static byte[] buildShort(RandomAccessibleInterval<ShortType> imgTensor)
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
		byte[] byteArr = new byte[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			short val = tensorCursor.get().get();
			byte[] arr = ByteBuffer.allocate(4).putShort(val).array();
			System.arraycopy(arr, 0, byteArr, cc * 4, 4);
			cc ++;
		}
		return byteArr;
    }

    /**
     * Creates a byte array from a {@link UnsignedShortType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     */
    private static byte[] buildUShort(RandomAccessibleInterval<UnsignedShortType> imgTensor)
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
		byte[] byteArr = new byte[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int val = tensorCursor.get().get();
			short shortval;
			if (val >= Math.pow(2, 15))
				shortval = (short) (val - Math.pow(2, 16));
			else 
				shortval = (short) val;
			byte[] arr = ByteBuffer.allocate(4).putShort(shortval).array();
			System.arraycopy(arr, 0, byteArr, cc * 4, 4);
			cc ++;
		}
		return byteArr;
    }

    /**
     * Creates a byte array from a {@link IntType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
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
     * Creates a byte array from a {@link UnsignedIntType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     */
    private static byte[] buildUInt(RandomAccessibleInterval<UnsignedIntType> imgTensor)
    {
    	Cursor<UnsignedIntType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<UnsignedIntType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<UnsignedIntType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 4;
		for (long ss : imgTensor.dimensionsAsLongArray()) {flatSize *= ss;}
		byte[] byteArr = new byte[(int) flatSize];
		int cc =  0;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long val = tensorCursor.get().get();
			int intval;
			if (val >= Math.pow(2, 31))
				intval = (int) (val - Math.pow(2, 32));
			else 
				intval = (int) val;
			byte[] arr = ByteBuffer.allocate(4).putInt(intval).array();
			System.arraycopy(arr, 0, byteArr, cc * 4, 4);
			cc ++;
		}
		return byteArr;
    }

    /**
     * Creates a byte array from a {@link FloatType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
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
     * Creates a byte array from a {@link DoubleType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
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
