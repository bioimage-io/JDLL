/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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
import java.util.HashMap;
import java.util.Map;

import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.Utils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.blocks.PrimitiveBlocks;
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

/**
 * Class that converts {@link RandomAccessibleInterval}-based JDLL tensors into
 * byte based maps that can be sent to the Bioengine for processing
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class BioengineTensor {
	
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
	protected final static String OBJECT_KEY = "_rtype";
	/**
	 * Value corresponding to the type of the array in the
	 * {@link #inputs} map
	 */
	protected static final String NDARRAY_VALUE = "ndarray";
	/**
	 * Value corresponding to a parameter
	 */
	private static final String PARAMETER_VALUE = "parameter";
	/**
	 * String key corresponding to the value of the array in the
	 * {@link #inputs} map
	 */
	protected static final String VALUE_KEY = "_rvalue";
	/**
	 * String key corresponding to the shape of the array in the
	 * {@link #inputs} map
	 */
	protected static final String SHAPE_KEY = "_rshape";
	/**
	 * String key corresponding to the dtype of the array in the
	 * {@link #inputs} map
	 */
	protected static final String DTYPE_KEY = "_rdtype";
	/**
	 * String used as tag for the float32 np dtype
	 */
	protected static final String FLOAT32_STR = "float32";
	/**
	 * String used as tag for the float64 np dtype
	 */
	protected static final String FLOAT64_STR = "float64";
	/**
	 * String used as tag for the byte or int8 np dtype
	 */
	protected static final String BYTE_STR = "int8";
	/**
	 * String used as tag for the byte or int16 np dtype
	 */
	protected static final String INT16_STR = "int16";
	/**
	 * String used as tag for the int32 np dtype
	 */
	protected static final String INT32_STR = "int32";
	/**
	 * String used as tag for the int64 np dtype
	 */
	protected static final String INT64_STR = "int64";
	/**
	 * String used as tag for the ubyte or uint8 np dtype
	 */
	protected static final String UBYTE_STR = "uint8";
	/**
	 * String used as tag for the uint16 np dtype
	 */
	protected static final String UINT16_STR = "uint16";
	/**
	 * String used as tag for the uint32 np dtype
	 */
	protected static final String UINT32_STR = "uint32";
	/**
	 * String used as tag for the boolean np dtype
	 */
	protected static final String BOOL_STR = "bool";

	/**
	 * Utility class.
	 */
	private BioengineTensor() {}

	/**
	 * Create an object that can be sent and understood by the bioengine.
	 * This method creates the needed object for a parameter in the bioengine
	 * 
	 * @param params
	 * 	the parameters that the model needs
	 * @return an object readable by the bioengine for parameters
	 */
	public static BioengineTensor build(Map<String, Object> params) {
		BioengineTensor bt = new BioengineTensor();
		bt.inputs.put(OBJECT_KEY, PARAMETER_VALUE);
		bt.inputs.put(SHAPE_KEY, new int[] {1});
		bt.inputs.put(DTYPE_KEY, "BYTES");
		bt.inputs.put(VALUE_KEY, params);
		return bt;
	}
	
	/**
	 * From a JDLL tensors, this method creates an object that contains 
	 * a map that can be sent to the bioengine.
	 * This method creates the needed object for an Image or ndarray
	 * 
	 * @param <T>
	 * 	ImgLib2 datatype that the tensor can have
	 * @param tensor
	 * 	the tensor containing all the needed data
	 * @return an object contianing the map that can be sent to the bioengine
	 */
	public static < T extends RealType< T > & NativeType< T > > 
				BioengineTensor build(Tensor<T> tensor) {
		return build(tensor, ByteOrder.LITTLE_ENDIAN);
	}
	
	/**
	 * From a JDLL tensors, this method creates an object that contains 
	 * a map that can be sent to the bioengine.
	 * This method creates the needed object for an Image or ndarray
	 * 
	 * @param <T>
	 * 	ImgLib2 datatype that the tensor can have
	 * @param tensor
	 * 	the tensor containing all the needed data
	 * @param order
	 * 	order of the bytes
	 * @return an object contianing the map that can be sent to the bioengine
	 */
	public static < T extends RealType< T > & NativeType< T > > 
				BioengineTensor build(Tensor<T> tensor, ByteOrder order) {
		BioengineTensor bt = new BioengineTensor();
		bt.inputs.put(OBJECT_KEY, NDARRAY_VALUE);
		bt.inputs.put(SHAPE_KEY, tensor.getShape());
		bt.inputs.put(VALUE_KEY, tensor.getShape());
		RandomAccessibleInterval<T> rai = tensor.getData();
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
			bt.inputs.put( VALUE_KEY, buildByte( Cast.unchecked( rai ) ) );
    		bt.inputs.put(DTYPE_KEY, BYTE_STR);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedByteType) {
			bt.inputs.put( VALUE_KEY, buildUByte( ( Cast.unchecked( rai ) ) ) );
    		bt.inputs.put(DTYPE_KEY, UBYTE_STR);
    	} else if (Util.getTypeFromInterval(rai) instanceof ShortType) {
			bt.inputs.put( VALUE_KEY, buildShort( Cast.unchecked( rai ), order ) );
    		bt.inputs.put(DTYPE_KEY, INT16_STR);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedShortType) {
			bt.inputs.put( VALUE_KEY, buildUShort( Cast.unchecked( rai ), order ) );
    		bt.inputs.put(DTYPE_KEY, UINT16_STR);
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
			bt.inputs.put( VALUE_KEY, buildInt( Cast.unchecked( rai ), order ) );
    		bt.inputs.put(DTYPE_KEY, INT32_STR);
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedIntType) {
			bt.inputs.put( VALUE_KEY, buildUInt( Cast.unchecked( rai ), order ) );
    		bt.inputs.put(DTYPE_KEY, UINT32_STR);
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
			bt.inputs.put( VALUE_KEY, buildFloat( Cast.unchecked( rai ), order ) );
    		bt.inputs.put(DTYPE_KEY, FLOAT32_STR);
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
			bt.inputs.put( VALUE_KEY, buildDouble( Cast.unchecked( rai ), order ) );
    		bt.inputs.put(DTYPE_KEY, FLOAT64_STR);
    	} else if (Util.getTypeFromInterval(rai) instanceof LongType) {
			bt.inputs.put( VALUE_KEY, buildLong( Cast.unchecked( rai ), order ) );
    		bt.inputs.put(DTYPE_KEY, INT64_STR);
    	} else {
            throw new IllegalArgumentException("The image has an unsupported type: " + Util.getTypeFromInterval(rai).getClass().toString());
    	}
		return bt;
	}
	
	/**
	 * 
	 * @return the map that is actually serialized and sent to the bioengine
	 */
	public Map<String, Object> getAsMap() {
		return this.inputs;
	}

	/**
	 * Create byte array from the backend {@link RandomAccessibleInterval} of a JDLL tensor
	 * @param <T>
	 * 	ImgLib2 datatype that the tensor can have
	 * @param tensor
	 * 	the tensor containing all the needed data
	 * @return the backend {@link RandomAccessibleInterval} of a JDLL tensor as a byte array,
	 *  does not contain information about dimensions or data type
	 */
	public static  < T extends RealType< T > & NativeType< T > >
				byte[] createByteArray(Tensor<T> tensor) {
		return imglib2ToByteArray(tensor.getData());
	}

	/**
	 * Create byte array from the backend {@link RandomAccessibleInterval} of a JDLL tensor
	 * @param <T>
	 * 	ImgLib2 datatype that the tensor can have
	 * @param tensor
	 * 	the tensor containing all the needed data
	 * @param order
	 * 	order of the bytes
	 * @return the backend {@link RandomAccessibleInterval} of a JDLL tensor as a byte array,
	 *  does not contain information about dimensions or data type
	 */
	public static  < T extends RealType< T > & NativeType< T > >
				byte[] createByteArray(Tensor<T> tensor, ByteOrder order) {
		return imglib2ToByteArray(tensor.getData(), order);
	}

	
	/**
	 * 
	 * Create byte array from a {@link RandomAccessibleInterval} 
	 * @param <T>
	 * 	ImgLib2 datatype that the image can have
	 * @param rai
	 * 	the image containing all the needed data
	 * @return the {@link RandomAccessibleInterval} as a byte array, does not contain information
	 * 	about dimensions or data type
	 */
	public static < T extends RealType< T > & NativeType< T > >
				byte[] imglib2ToByteArray(RandomAccessibleInterval<T> rai) {
		return imglib2ToByteArray(rai, ByteOrder.LITTLE_ENDIAN);
	}

	
	/**
	 * 
	 * Create byte array from a {@link RandomAccessibleInterval} 
	 * @param <T>
	 * 	ImgLib2 datatype that the image can have
	 * @param rai
	 * 	the image containing all the needed data
     * @param order
     * 	order of the bytes
	 * @return the {@link RandomAccessibleInterval} as a byte array, does not contain information
	 * 	about dimensions or data type
	 */
	public static < T extends RealType< T > & NativeType< T > >
				byte[] imglib2ToByteArray(RandomAccessibleInterval<T> rai, ByteOrder order) {
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
			return buildByte( Cast.unchecked( rai ) );
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedByteType) {
			return buildUByte( Cast.unchecked( rai ) );
    	} else if (Util.getTypeFromInterval(rai) instanceof ShortType) {
			return buildShort( Cast.unchecked( rai ), order );
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedShortType) {
			return buildUShort( Cast.unchecked( rai ), order );
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
			return buildInt( Cast.unchecked( rai ), order );
    	} else if (Util.getTypeFromInterval(rai) instanceof UnsignedIntType) {
			return buildUInt( Cast.unchecked( rai ), order );
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
			return buildFloat( Cast.unchecked( rai ), order );
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
			return buildDouble( Cast.unchecked( rai ), order );
    	} else if (Util.getTypeFromInterval(rai) instanceof LongType) {
			return buildLong( Cast.unchecked( rai ), order );
    	} else {
            throw new IllegalArgumentException("The image has an unsupported type: " + Util.getTypeFromInterval(rai).getClass().toString());
    	}
	}

    /**
     * Creates a byte array from a {@link ByteType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param order
     * 	order of the bytes
     * @return byte array containing the data of the image as a flat byte array
     */
    private static byte[] buildByte(RandomAccessibleInterval<ByteType> imgTensor)
    {
		imgTensor = Utils.transpose(imgTensor);
		PrimitiveBlocks< ByteType > blocks = PrimitiveBlocks.of( imgTensor );
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final byte[] flatArr = new byte[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( imgTensor.minAsLongArray(), flatArr, sArr );
		return flatArr;
    }

    /**
     * Creates a byte array from a {@link UnsignedByteType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param order
     * 	order of the bytes
     * @return byte array containing the data of the image as a flat byte array
     */
    private static byte[] buildUByte(RandomAccessibleInterval<UnsignedByteType> imgTensor)
    {
		imgTensor = Utils.transpose(imgTensor);
		PrimitiveBlocks< UnsignedByteType > blocks = PrimitiveBlocks.of( imgTensor );
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final byte[] flatArr = new byte[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( imgTensor.minAsLongArray(), flatArr, sArr );
		return flatArr;
    }

    /**
     * Creates a byte array from a {@link ShortType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param order
     * 	order of the bytes
     * @return byte array containing the data of the image as a flat byte array
     */
    private static byte[] buildShort(RandomAccessibleInterval<ShortType> imgTensor, ByteOrder order)
    {
		imgTensor = Utils.transpose(imgTensor);
		PrimitiveBlocks< ShortType > blocks = PrimitiveBlocks.of( imgTensor );
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final short[] flatArr = new short[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( imgTensor.minAsLongArray(), flatArr, sArr );
		ByteBuffer buffer = ByteBuffer.allocate(flatArr.length * 2).order(order);
        for (short d : flatArr) {
            buffer.putShort(d);
        }
		return buffer.array();
    }

    /**
     * Creates a byte array from a {@link UnsignedShortType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param order
     * 	order of the bytes
     * @return byte array containing the data of the image as a flat byte array
     */
    private static byte[] buildUShort(RandomAccessibleInterval<UnsignedShortType> imgTensor, ByteOrder order)
    {
		imgTensor = Utils.transpose(imgTensor);
		PrimitiveBlocks< UnsignedShortType > blocks = PrimitiveBlocks.of( imgTensor );
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final short[] flatArr = new short[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( imgTensor.minAsLongArray(), flatArr, sArr );
		ByteBuffer buffer = ByteBuffer.allocate(flatArr.length * 2).order(order);
        for (short d : flatArr) {
            buffer.putShort(d);
        }
		return buffer.array();
    }

    /**
     * Creates a byte array from a {@link IntType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param order
     * 	order of the bytes
     * @return byte array containing the data of the image as a flat byte array
     */
    private static byte[] buildInt(RandomAccessibleInterval<IntType> imgTensor, ByteOrder order)
    {
		imgTensor = Utils.transpose(imgTensor);
		PrimitiveBlocks< IntType > blocks = PrimitiveBlocks.of( imgTensor );
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final int[] flatArr = new int[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( imgTensor.minAsLongArray(), flatArr, sArr );
		ByteBuffer buffer = ByteBuffer.allocate(flatArr.length * 4).order(order);
        for (int d : flatArr) {
            buffer.putInt(d);
        }
		return buffer.array();
    }

    /**
     * Creates a byte array from a {@link UnsignedIntType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param order
     * 	order of the bytes
     * @return byte array containing the data of the image as a flat byte array
     */
    private static byte[] buildUInt(RandomAccessibleInterval<UnsignedIntType> imgTensor, ByteOrder order)
    {
		imgTensor = Utils.transpose(imgTensor);
		PrimitiveBlocks< UnsignedIntType > blocks = PrimitiveBlocks.of( imgTensor );
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final int[] flatArr = new int[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( imgTensor.minAsLongArray(), flatArr, sArr );
		ByteBuffer buffer = ByteBuffer.allocate(flatArr.length * 4).order(order);
        for (int d : flatArr) {
            buffer.putInt(d);
        }
		return buffer.array();
    }

    /**
     * Creates a byte array from a {@link LongType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param order
     * 	order of the bytes
     * @return byte array containing the data of the image as a flat byte array
     */
    private static byte[] buildLong(RandomAccessibleInterval<LongType> imgTensor, ByteOrder order)
    {
		imgTensor = Utils.transpose(imgTensor);
		PrimitiveBlocks< LongType > blocks = PrimitiveBlocks.of( imgTensor );
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final long[] flatArr = new long[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( imgTensor.minAsLongArray(), flatArr, sArr );
		ByteBuffer buffer = ByteBuffer.allocate(flatArr.length * 8).order(order);
        for (long d : flatArr) {
            buffer.putLong(d);
        }
		return buffer.array();
    }

    /**
     * Creates a byte array from a {@link FloatType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param order
     * 	order of the bytes
     * @return byte array containing the data of the image as a flat byte array
     */
    private static byte[] buildFloat(RandomAccessibleInterval<FloatType> imgTensor, ByteOrder order)
    {
		imgTensor = Utils.transpose(imgTensor);
		PrimitiveBlocks< FloatType > blocks = PrimitiveBlocks.of( imgTensor );
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final float[] flatArr = new float[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( imgTensor.minAsLongArray(), flatArr, sArr );
		ByteBuffer buffer = ByteBuffer.allocate(flatArr.length * 4).order(order);
        for (float d : flatArr) {
            buffer.putFloat(d);
        }
		return buffer.array();
    }

    /**
     * Creates a byte array from a {@link DoubleType} {@link RandomAccessibleInterval}.
     * 
     * @param imgTensor 
     * 	{@link RandomAccessibleInterval} to be mapped into byte buffer
     * @param order
     * 	order of the bytes
     * @return byte array containing the data of the image as a flat byte array
     */
    private static byte[] buildDouble(RandomAccessibleInterval<DoubleType> imgTensor, ByteOrder order)
    {
		imgTensor = Utils.transpose(imgTensor);
		PrimitiveBlocks< DoubleType > blocks = PrimitiveBlocks.of( imgTensor );
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		int size = 8;
		for (long ll : tensorShape) size *= ll;
		final double[] flatArr = new double[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( imgTensor.minAsLongArray(), flatArr, sArr );
		ByteBuffer buffer = ByteBuffer.allocate(flatArr.length * 8).order(order);
        for (double d : flatArr) {
            buffer.putDouble(d);
        }
		return buffer.array();
    }
}
