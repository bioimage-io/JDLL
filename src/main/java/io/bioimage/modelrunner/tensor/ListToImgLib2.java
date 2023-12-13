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

import java.io.ByteArrayOutputStream;
import java.math.BigDecimal;
import java.util.List;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
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

/**
 * A {@link RandomAccessibleInterval} and {@link Tensor} builder from {@link List} objects.
 * This class was originally designed to be used with the Appose library to favor the inter-processing
 * communication with the Python process using **pipes**.
 * The arrays sent from java to Python where decoded into Java {@link List}s. This class is useful to convert
 * those flat lists into {@link RandomAccessibleInterval}s or {@link Tensor}s given other details as the shape.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class ListToImgLib2 {

    /**
     * Not used (Utility class).
     */
    private ListToImgLib2()
    {
    }

    /**
     * Creates a {@link Tensor} from the information stored in a {@link List} given the shape of the tensor,
     * its axes, the data type and the name
     * 
     * @param <T>
     * 	the type of the generated tensor
     * @param array
     * 	{@link List} object containing the flat data of the tensor. The data type should be either the same as
     * 	provided in the 'dtype' argument ({@link Byte}, {@link Short}, {@link Integer}, {@link Long},
     * 	{@link Float}, {@link Double}). The List can also be of type {@link Byte} and the argument 'dtype' can be any type
     * 	as longg as the bytes of the list encode the corresponding data type. For example if the dtype is 'int32' there should
     * 	be 4 times more elements than what is required by the shape because each 4 bytes corresponds to one int32 value.
     * @param shape
     * 	{@link List} containing the shape of the tensor that wants to be created from the flat array
     * @param axes
     * 	String containing the axes order. Which dimension correspond to which axes
     * @param dtype 
     * 	data type of the tensor that is going to be reconstructed from the flat data array
     * @param name
     * 	name of the tensor
     * @return the reconstructed tensor
     * @throws IllegalArgumentException if the data type defined by the argument 'dtype' is not supported
     * 	or if the data type of the components in the List of the  'array' argument is not valid
     */
    @SuppressWarnings("unchecked")
    public static < T extends RealType< T > & NativeType< T > > Tensor<T> 
    		buildTensor(List<?> array, List<Integer> shape, String axes, String dtype, String name) 
    				throws IllegalArgumentException
    {
		return Tensor.build(name, axes, (RandomAccessibleInterval<T>) build(array, shape, dtype));
    }

    /**
     * Creates a {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI
     * and the data type 
     * 
     * @param <T>
     * 	the type of the generated {@link RandomAccessibleInterval}
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}. The data type should be either the same as
     * 	provided in the 'dtype' argument ({@link Byte}, {@link Short}, {@link Integer}, {@link Long},
     * 	{@link Float}, {@link Double}). The List can also be of type {@link Byte} and the argument 'dtype' can be any type
     * 	as long as the bytes of the list encode the corresponding data type. For example if the dtype is 'int32' there should
     * 	be 4 times more elements than what is required by the shape because each 4 bytes corresponds to one int32 value.
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @param dtype 
     * 	data type of the tensor that is going to be reconstructed from the flat data array
     * @return the reconstructed {@link RandomAccessibleInterval}
     * @throws IllegalArgumentException if the data type defined by the argument 'dtype' is not supported
     * 	or if the data type of the components in the List of the  'array' argument is not valid
     */
    @SuppressWarnings("unchecked")
    public static < T extends RealType< T > & NativeType< T > > RandomAccessibleInterval<T> build(List<?> array, List<Integer> shape, String dtype) throws IllegalArgumentException
    {
    	if (shape.size() == 0 || array.size() == 0)
    		return null;
    	
        Img<T> data;
		switch (dtype)
        {
	    	case "int8":
	            data = Cast.unchecked(buildInt8(array, shape));
	            break;
	    	case "uint8":
	            data = Cast.unchecked(buildUint8(array, shape));
	            break;
	    	case "int16":
	            data = Cast.unchecked(buildInt16(array, shape));
	            break;
	    	case "uint16":
	            data = Cast.unchecked(buildUint16(array, shape));
	            break;
            case "int32":
            	data = Cast.unchecked(buildInt32(array, shape));
                break;
            case "uint32":
            	data = Cast.unchecked(buildUint32(array, shape));
                break;
            case "int64":
            	data = Cast.unchecked(buildInt64(array, shape));
                break;
            case "float32":
            	data = Cast.unchecked(buildFloat32(array, shape));
                break;
            case "float64":
            	data = Cast.unchecked(buildFloat64(array, shape));
                break;
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + dtype);
        }
		return data;
    }

    /**
     * Creates a {@link ByteType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain either {@link Byte} objects or {@link Integer} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     * @throws IllegalArgumentException if the input argument 'array' is not a {@link List} of {@link Byte}s or {@link Integer}s
     */
    private static RandomAccessibleInterval<ByteType> buildInt8(List array, List<Integer> tensorShape) throws IllegalArgumentException
    {
    	if (!(array.get(0) instanceof Byte)
        		&& !(array.get(0) instanceof Integer)) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'int8' using Java array of class: " + array.get(0).getClass());
    	} else if (array.get(0) instanceof Integer) {
    		return buildInt8FromInteger((List<Integer>) array, tensorShape);
    	} else {
    		return buildInt8FromByte((List<Byte>) array, tensorShape);
    	}
	}

    /**
     * Creates a {@link ByteType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Byte} objects 
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<ByteType> buildInt8FromByte(List<Byte> tensor, List<Integer> tensorShape)
    {
    	ByteArrayOutputStream baos = new ByteArrayOutputStream(tensor.size());
    	tensor.forEach(baos::write);
    	long[] shape = new long[tensorShape.size()];
    	RandomAccessibleInterval<ByteType> rai = ArrayImgs.bytes(baos.toByteArray(), shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link ByteType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Integer} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<ByteType> buildInt8FromInteger(List<Integer> tensor, List<Integer> tensorShape)
    {
    	byte[] bytes = new byte[tensor.size()];
    	int c = 0;
    	for (Integer it : tensor)
    		bytes[c ++] = it.byteValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<ByteType> rai = ArrayImgs.bytes(bytes, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link UnsignedByteType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain either {@link Byte} objects or {@link Integer} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     * @throws IllegalArgumentException if the input argument 'array' is not a {@link List} of {@link Byte}s or {@link Integer}s
     */
    private static RandomAccessibleInterval<UnsignedByteType> buildUint8(List array, List<Integer> tensorShape) throws IllegalArgumentException
    {
    	if (!(array.get(0) instanceof Byte)
        		&& !(array.get(0) instanceof Integer)) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'int8' using Java array of class: " + array.get(0).getClass());
    	} else if (array.get(0) instanceof Byte) {
    		return buildUint8FromByte((List<Byte>) array, tensorShape);
    	} else {
    		return buildUint8FromInteger((List<Integer>) array, tensorShape);
    	}
	}

    /**
     * Creates a {@link UnsignedByteType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Byte} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<UnsignedByteType> buildUint8FromByte(List<Byte> tensor, List<Integer> tensorShape)
    {
    	byte[] bytes = new byte[tensor.size()];
    	int c = 0;
    	for (Byte it : tensor)
    		bytes[c ++] = it.byteValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<UnsignedByteType> rai = ArrayImgs.unsignedBytes(bytes, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link UnsignedByteType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Integer} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<UnsignedByteType> buildUint8FromInteger(List<Integer> tensor, List<Integer> tensorShape)
    {
    	byte[] bytes = new byte[tensor.size()];
    	int c = 0;
    	for (Integer it : tensor)
    		bytes[c ++] = it.byteValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<UnsignedByteType> rai = ArrayImgs.unsignedBytes(bytes, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link ShortType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain either {@link Short} objects or {@link Integer} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     * @throws IllegalArgumentException if the input argument 'array' is not a {@link List} of {@link Short}s or {@link Integer}s
     */
    private static RandomAccessibleInterval<ShortType> buildInt16(List array, List<Integer> tensorShape) throws IllegalArgumentException
    {
    	if (!(array.get(0) instanceof Short)
        		&& !(array.get(0) instanceof Integer)) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'int16' using Java array of class: " + array.get(0).getClass());
    	} else if (array.get(0) instanceof Short) {
    		return buildInt16FromShort((List<Short>) array, tensorShape);
    	} else {
    		return buildInt16FromInteger((List<Integer>) array, tensorShape);
    	}
	}

    /**
     * Creates a {@link ShortType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Short} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<ShortType> buildInt16FromShort(List<Short> tensor, List<Integer> tensorShape)
    {
    	short[] shorts = new short[tensor.size()];
    	int c = 0;
    	for (Short it : tensor)
    		shorts[c ++] = it.shortValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<ShortType> rai = ArrayImgs.shorts(shorts, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link ShortType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Integer} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<ShortType> buildInt16FromInteger(List<Integer> tensor, List<Integer> tensorShape)
    {
    	short[] shorts = new short[tensor.size()];
    	int c = 0;
    	for (Integer it : tensor)
    		shorts[c ++] = it.shortValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<ShortType> rai = ArrayImgs.shorts(shorts, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link UnsignedShortType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain either {@link Short} objects or {@link Integer} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     * @throws IllegalArgumentException if the input argument 'array' is not a {@link List} of {@link Short}s or {@link Integer}s
     */
    private static RandomAccessibleInterval<UnsignedShortType> buildUint16(List array, List<Integer> tensorShape) throws IllegalArgumentException
    {
    	if (!(array.get(0) instanceof Short)
        		&& !(array.get(0) instanceof Integer)) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'uint16' using Java array of class: " + array.get(0).getClass());
    	} else if (array.get(0) instanceof Short) {
    		return buildUint16FromShort((List<Short>) array, tensorShape);
    	} else {
    		return buildUint16FromInteger((List<Integer>) array, tensorShape);
    	}
	}

    /**
     * Creates a {@link UnsignedShortType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Short} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<UnsignedShortType> buildUint16FromShort(List<Short> tensor, List<Integer> tensorShape)
    {
    	short[] shorts = new short[tensor.size()];
    	int c = 0;
    	for (Short it : tensor)
    		shorts[c ++] = it.shortValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<UnsignedShortType> rai = ArrayImgs.unsignedShorts(shorts, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link UnsignedShortType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Integer} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<UnsignedShortType> buildUint16FromInteger(List<Integer> tensor, List<Integer> tensorShape)
    {
    	short[] shorts = new short[tensor.size()];
    	int c = 0;
    	for (Integer it : tensor)
    		shorts[c ++] = it.shortValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<UnsignedShortType> rai = ArrayImgs.unsignedShorts(shorts, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link IntType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Integer} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     * @throws IllegalArgumentException if the input argument 'array' is not a {@link List} of @link Integer}s
     */
    private static RandomAccessibleInterval<IntType> buildInt32(List array, List<Integer> tensorShape) throws IllegalArgumentException
    {
    	if (!(array.get(0) instanceof Integer)) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'int32' using Java array of class: " + array.get(0).getClass());
    	} else {
    		return buildInt32FromInteger((List<Integer>) array, tensorShape);
    	}
	}

    /**
     * Creates a {@link IntType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Integer} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<IntType> buildInt32FromInteger(List<Integer> tensor, List<Integer> tensorShape)
    {
    	int[] ints = new int[tensor.size()];
    	int c = 0;
    	for (Integer it : tensor)
    		ints[c ++] = it.intValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<IntType> rai = ArrayImgs.ints(ints, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link UnsignedIntType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain either {@link Long} objects or {@link Integer} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     * @throws IllegalArgumentException if the input argument 'array' is not a {@link List} of {@link Long}s or {@link Integer}s
     */
    private static RandomAccessibleInterval<UnsignedIntType> buildUint32(List array, List<Integer> tensorShape) throws IllegalArgumentException
    {
    	if (!(array.get(0) instanceof Long)
        		&& !(array.get(0) instanceof Integer)) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'uint32' using Java array of class: " + array.get(0).getClass());
    	} else if (array.get(0) instanceof Long) {
    		return buildUint32FromLong((List<Long>) array, tensorShape);
    	} else {
    		return buildUint32FromInteger((List<Integer>) array, tensorShape);
    	}
	}

    /**
     * Creates a {@link UnsignedIntType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Integer} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<UnsignedIntType> buildUint32FromInteger(List<Integer> tensor, List<Integer> tensorShape)
    {
    	int[] ints = new int[tensor.size()];
    	int c = 0;
    	for (Integer it : tensor)
    		ints[c ++] = it.intValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<UnsignedIntType> rai = ArrayImgs.unsignedInts(ints, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link UnsignedIntType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Long} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<UnsignedIntType> buildUint32FromLong(List<Long> tensor, List<Integer> tensorShape)
    {
    	int[] ints = new int[tensor.size()];
    	int c = 0;
    	for (Long it : tensor)
    		ints[c ++] = it.intValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<UnsignedIntType> rai = ArrayImgs.unsignedInts(ints, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link LongType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain either {@link Long} objects or {@link Integer} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     * @throws IllegalArgumentException if the input argument 'array' is not a {@link List} of {@link Long}s or {@link Integer}s
     */
    private static RandomAccessibleInterval<LongType> buildInt64(List array, List<Integer> tensorShape) throws IllegalArgumentException
    {
    	if (!(array.get(0) instanceof Long)
        		&& !(array.get(0) instanceof Integer)) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'int64' using Java array of class: " + array.get(0).getClass());
    	} else if (array.get(0) instanceof Long) {
    		return buildInt64FromLong((List<Long>) array, tensorShape);
    	} else {
    		return buildInt64FromInteger((List<Integer>) array, tensorShape);
    	}
	}

    /**
     * Creates a {@link LongType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Long} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<LongType> buildInt64FromLong(List<Long> tensor, List<Integer> tensorShape)
    {
    	long[] longs = new long[tensor.size()];
    	int c = 0;
    	for (Long it : tensor)
    		longs[c ++] = it.longValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<LongType> rai = ArrayImgs.longs(longs, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link LongType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Integer} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<LongType> buildInt64FromInteger(List<Integer> tensor, List<Integer> tensorShape)
    {
    	long[] longs = new long[tensor.size()];
    	int c = 0;
    	for (Integer it : tensor)
    		longs[c ++] = it.longValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<LongType> rai = ArrayImgs.longs(longs, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link FloatType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain either {@link Float} objects or {@link BigDecimal} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     * @throws IllegalArgumentException if the input argument 'array' is not a {@link List} of {@link Float}s or {@link BigDecimal}s
     */
    private static RandomAccessibleInterval<FloatType> buildFloat32(List array, List<Integer> tensorShape) throws IllegalArgumentException
    {
    	if (!(array.get(0) instanceof Float)
    			&& !(array.get(0) instanceof BigDecimal)) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'float32' using Java array of class: " + array.get(0).getClass());
    	} else if (array.get(0) instanceof Float) {
    		return buildFloat32FromFloat((List<Float>) array, tensorShape);
    	} else {
    		return buildFloat32FromBigDecimal((List<BigDecimal>) array, tensorShape);
    	}
	}

    /**
     * Creates a {@link FloatType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Float} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<FloatType> buildFloat32FromFloat(List<Float> tensor, List<Integer> tensorShape)
    {
    	float[] floats = new float[tensor.size()];
    	int c = 0;
    	for (Float it : tensor)
    		floats[c ++] = it.floatValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<FloatType> rai = ArrayImgs.floats(floats, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link FloatType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link BigDecimal} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<FloatType> buildFloat32FromBigDecimal(List<BigDecimal> tensor, List<Integer> tensorShape)
    {
    	float[] floats = new float[tensor.size()];
    	int c = 0;
    	for (BigDecimal it : tensor)
    		floats[c ++] = it.floatValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<FloatType> rai = ArrayImgs.floats(floats, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link DoubleType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain either {@link Float} objects, {@link BigDecimal} objects or {@link Double} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     * @throws IllegalArgumentException if the input argument 'array' is not a {@link List} of {@link Float}s , {@link Double}s or {@link BigDecimal}s
     */
    private static RandomAccessibleInterval<DoubleType> buildFloat64(List array, List<Integer> tensorShape) throws IllegalArgumentException
    {
    	if (!(array.get(0) instanceof Float)
        		&& !(array.get(0) instanceof Double)
        		&& !(array.get(0) instanceof BigDecimal)) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'float64' using Java array of class: " + array.get(0).getClass());
    	} else if (array.get(0) instanceof Float) {
    		return buildFloat64FromFloat((List<Float>) array, tensorShape);
    	} else if (array.get(0) instanceof Double) {
    		return buildFloat64FromDouble((List<Double>) array, tensorShape);
    	} else {
    		return buildFloat64FromBigDecimal((List<BigDecimal>) array, tensorShape);
    	}
	}

    /**
     * Creates a {@link DoubleType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Double} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<DoubleType> buildFloat64FromDouble(List<Double> tensor, List<Integer> tensorShape)
    {
    	double[] doubles = new double[tensor.size()];
    	int c = 0;
    	for (Double it : tensor)
    		doubles[c ++] = it.byteValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<DoubleType> rai = ArrayImgs.doubles(doubles, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link DoubleType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link Float} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<DoubleType> buildFloat64FromFloat(List<Float> tensor, List<Integer> tensorShape)
    {
    	double[] doubles = new double[tensor.size()];
    	int c = 0;
    	for (Float it : tensor)
    		doubles[c ++] = it.doubleValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<DoubleType> rai = ArrayImgs.doubles(doubles, shape);
		return Utils.transpose(rai);
	}

    /**
     * Creates a {@link DoubleType} {@link RandomAccessibleInterval} from the information stored in a {@link List} given the shape of the RAI.
     * 
     * @param array
     * 	{@link List} object containing the flat data of the {@link RandomAccessibleInterval}.
     * 	The list should contain {@link BigDecimal} objects
     * @param shape
     * 	{@link List} containing the shape of the {@link RandomAccessibleInterval} that wants to be created from the flat array
     * @return the reconstructed {@link RandomAccessibleInterval}
     */
    private static RandomAccessibleInterval<DoubleType> buildFloat64FromBigDecimal(List<BigDecimal> tensor, List<Integer> tensorShape)
    {
    	double[] doubles = new double[tensor.size()];
    	int c = 0;
    	for (BigDecimal it : tensor)
    		doubles[c ++] = it.byteValue();
    	long[] shape = new long[tensorShape.size()];
    	c = 0;
    	for (Integer it : tensorShape)
    		shape[c ++] = it.longValue();
    	RandomAccessibleInterval<DoubleType> rai = ArrayImgs.doubles(doubles, shape);
		return Utils.transpose(rai);
	}
}
