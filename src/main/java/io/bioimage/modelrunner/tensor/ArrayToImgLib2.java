/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java API for Tensorflow 1.
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

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.IndexingUtils;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * A {@link Img} builder from {@link ByteBuffer} objects
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class ArrayToImgLib2 {

    /**
     * Not used (Utility class).
     */
    private ArrayToImgLib2()
    {
    }

    /**
     * Creates a {@link Tensor} from the information stored in a {@link ByteBuffer}
     * 
     * @param <T>
     * 	the type of the generated tensor
     * @param buff
     * 	byte buffer to get the tensor info from
     * @return the tensor generated from the bytebuffer
     * @throws IllegalArgumentException if the data type of the tensor saved in the bytebuffer is
     * not supported
     */
    @SuppressWarnings("unchecked")
    public static < T extends RealType< T > & NativeType< T > > Tensor<T> 
    		buildTensor(Object array, Integer[] shape, String axes, String dtype, String name) 
    				throws IllegalArgumentException
    {
		return Tensor.build(name, axes, (RandomAccessibleInterval<T>) build(array, shape, dtype));
    }

    /**
     * Creates a {@link Img} from the information stored in a {@link ByteBuffer}
     * 
     * @param <T>
     * 	data type of the image
     * @param byteBuff
     *        The bytebyuffer that contains info to create a tenosr or a {@link Img}
     * @return The imglib2 image {@link Img} built from the bytebuffer info.
     * @throws IllegalArgumentException if the data type of the tensor saved in the bytebuffer is
     * not supported
     */
    @SuppressWarnings("unchecked")
    public static <T extends Type<T>> Img<T> build(Object array, Integer[] shape, String dtype) throws IllegalArgumentException
    {
    	if (!array.getClass().isArray()) {
    		throw new IllegalArgumentException("");
    	}
    	if (shape.length == 0)
    		return null;
    	
        Img<?> data;
		switch (dtype)
        {
	    	case "int8":
	            data = (Img<?>) buildInt8(array, shape);
	            break;
	    	case "uint8":
	            data = (Img<?>) buildUint8(array, shape);
	            break;
	    	case "int16":
	            data = (Img<?>) buildInt16(array, shape);
	            break;
	    	case "uint16":
	            data = (Img<?>) buildUint16(array, shape);
	            break;
            case "int32":
            	data = (Img<?>) buildInt32(array, shape);
                break;
            case "uint32":
            	data = (Img<?>) buildUint32(array, shape);
                break;
            case "int64":
            	data = (Img<?>) buildInt64(array, shape);
                break;
            case "float16":
            	data = (Img<?>) buildFloat16(array, shape);
                break;
            case "float32":
            	data = (Img<?>) buildFloat32(array, shape);
                break;
            case "float64":
            	data = (Img<?>) buildFloat64(array, shape);
                break;
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + dtype);
        }
		return data;
    }

    /**
     * Builds a ByteType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<ByteType> buildInt8(Object array, Integer[] tensorShape)
    {
    	if (!array.getClass().getComponentType().equals(byte.class)
        		&& !Byte.class.isAssignableFrom(array.getClass().getComponentType())
        		&& !Integer.class.isAssignableFrom(array.getClass().getComponentType())) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'int8' using Java array of class: " + array.getClass().getComponentType());
    	} else if (array.getClass().getComponentType().equals(byte.class)) {
    		return buildInt8((byte[]) array, tensorShape);
    	} else if (Byte.class.isAssignableFrom(array.getClass().getComponentType())) {
    		return buildInt8((Byte[]) array, tensorShape);
    	} else {
    		return buildInt8((Integer[]) array, tensorShape);
    	}
	}

    /**
     * Builds a ByteType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<ByteType> buildInt8(Byte[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< ByteType > factory = new ArrayImgFactory<>( new ByteType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(shape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i]);
		}
	 	return outputImg;
	}

    /**
     * Builds a ByteType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<ByteType> buildInt8(Integer[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< ByteType > factory = new ArrayImgFactory<>( new ByteType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(shape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i].byteValue());
		}
	 	return outputImg;
	}

    /**
     * Builds a ByteType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<ByteType> buildInt8(byte[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< ByteType > factory = new ArrayImgFactory<>( new ByteType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(shape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i]);
		}
	 	return outputImg;
	}

    /**
     * Builds a ShortType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<ShortType> buildInt16(Object array, Integer[] tensorShape)
    {
    	if (!array.getClass().getComponentType().equals(short.class)
        		&& !Short.class.isAssignableFrom(array.getClass().getComponentType())
        		&& !Integer.class.isAssignableFrom(array.getClass().getComponentType())) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'int16' using Java array of class: " + array.getClass().getComponentType());
    	} else if (array.getClass().getComponentType().equals(short.class)) {
    		return buildInt16((short[]) array, tensorShape);
    	} else if (Short.class.isAssignableFrom(array.getClass().getComponentType())) {
    		return buildInt16((Short[]) array, tensorShape);
    	} else {
    		return buildInt16((Integer[]) array, tensorShape);
    	}
	}

    /**
     * Builds a IntType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<ShortType> buildInt16(Short[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< ShortType > factory = new ArrayImgFactory<>( new ShortType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< ShortType > outputImg = (Img<ShortType>) factory.create(shape);
    	Cursor<ShortType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i]);
		}
	 	return outputImg;
	}

    /**
     * Builds a IntType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<ShortType> buildInt16(Integer[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< ShortType > factory = new ArrayImgFactory<>( new ShortType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< ShortType > outputImg = (Img<ShortType>) factory.create(shape);
    	Cursor<ShortType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i].shortValue());
		}
	 	return outputImg;
	}

    /**
     * Builds a IntType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<ShortType> buildInt16(short[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< ShortType > factory = new ArrayImgFactory<>( new ShortType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< ShortType > outputImg = (Img<ShortType>) factory.create(shape);
    	Cursor<ShortType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i]);
		}
	 	return outputImg;
	}

    /**
     * Builds a ByteType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<IntType> buildInt32(Object array, Integer[] tensorShape)
    {
    	if (!array.getClass().getComponentType().equals(int.class)
        		&& !Integer.class.isAssignableFrom(array.getClass().getComponentType())) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'int32' using Java array of class: " + array.getClass().getComponentType());
    	} else if (array.getClass().getComponentType().equals(int.class)) {
    		return buildInt32((int[]) array, tensorShape);
    	} else {
    		return buildInt32((Integer[]) array, tensorShape);
    	}
	}

    /**
     * Builds a IntType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<IntType> buildInt32(Integer[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< IntType > factory = new ArrayImgFactory<>( new IntType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< IntType > outputImg = (Img<IntType>) factory.create(shape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i]);
		}
	 	return outputImg;
	}

    /**
     * Builds a IntType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<IntType> buildInt32(int[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< IntType > factory = new ArrayImgFactory<>( new IntType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< IntType > outputImg = (Img<IntType>) factory.create(shape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i]);
		}
	 	return outputImg;
	}

    /**
     * Builds a ByteType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<UnsignedIntType> buildUint32(Object array, Integer[] tensorShape)
    {
    	if (!array.getClass().getComponentType().equals(int.class)
        		&& !Integer.class.isAssignableFrom(array.getClass().getComponentType())
        		&& !Long.class.isAssignableFrom(array.getClass().getComponentType())
        		&& !array.getClass().getComponentType().equals(long.class)) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'uint32' using Java array of class: " + array.getClass().getComponentType());
    	} else if (array.getClass().getComponentType().equals(int.class)) {
    		return buildUint32((int[]) array, tensorShape);
    	} else if (array.getClass().getComponentType().equals(long.class)) {
    		return buildUint32((long[]) array, tensorShape);
    	} else if (Long.class.isAssignableFrom(array.getClass().getComponentType())) {
    		return buildUint32((Long[]) array, tensorShape);
    	} else {
    		return buildUint32((Integer[]) array, tensorShape);
    	}
	}

    /**
     * Builds a IntType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<UnsignedIntType> buildUint32(Integer[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< UnsignedIntType > factory = new ArrayImgFactory<>( new UnsignedIntType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< UnsignedIntType > outputImg = (Img<UnsignedIntType>) factory.create(shape);
    	Cursor<UnsignedIntType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i]);
		}
	 	return outputImg;
	}

    /**
     * Builds a IntType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<UnsignedIntType> buildUint32(int[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< UnsignedIntType > factory = new ArrayImgFactory<>( new UnsignedIntType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< UnsignedIntType > outputImg = (Img<UnsignedIntType>) factory.create(shape);
    	Cursor<UnsignedIntType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i]);
		}
	 	return outputImg;
	}

    /**
     * Builds a IntType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<UnsignedIntType> buildUint32(Long[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< UnsignedIntType > factory = new ArrayImgFactory<>( new UnsignedIntType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< UnsignedIntType > outputImg = (Img<UnsignedIntType>) factory.create(shape);
    	Cursor<UnsignedIntType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i]);
		}
	 	return outputImg;
	}

    /**
     * Builds a IntType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<UnsignedIntType> buildUint32(long[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< UnsignedIntType > factory = new ArrayImgFactory<>( new UnsignedIntType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< UnsignedIntType > outputImg = (Img<UnsignedIntType>) factory.create(shape);
    	Cursor<UnsignedIntType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i]);
		}
	 	return outputImg;
	}

    /**
     * Builds a LongType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<LongType> buildInt64(Object array, Integer[] tensorShape)
    {
    	if (!array.getClass().getComponentType().equals(long.class)
        		&& !Long.class.isAssignableFrom(array.getClass().getComponentType())
        		&& !Integer.class.isAssignableFrom(array.getClass().getComponentType())) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'int64' using Java array of class: " + array.getClass().getComponentType());
    	} else if (array.getClass().getComponentType().equals(long.class)) {
    		return buildInt64((long[]) array, tensorShape);
    	} else if (Long.class.isAssignableFrom(array.getClass().getComponentType())) {
    		return buildInt64((Long[]) array, tensorShape);
    	} else {
    		return buildInt64((Integer[]) array, tensorShape);
    	}
	}

    /**
     * Builds a LongType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<LongType> buildInt64(Long[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< LongType > factory = new ArrayImgFactory<>( new LongType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< LongType > outputImg = (Img<LongType>) factory.create(shape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i]);
		}
	 	return outputImg;
	}

    /**
     * Builds a LongType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<LongType> buildInt64(Integer[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< LongType > factory = new ArrayImgFactory<>( new LongType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< LongType > outputImg = (Img<LongType>) factory.create(shape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i].longValue());
		}
	 	return outputImg;
	}

    /**
     * Builds a LongType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<LongType> buildInt64(long[] tensor, Integer[] tensorShape)
    {
    	final ArrayImgFactory< LongType > factory = new ArrayImgFactory<>( new LongType() );
    	long[] shape = IntStream.range(0, tensorShape.length).mapToLong(i -> tensorShape[i]).toArray();
        final Img< LongType > outputImg = (Img<LongType>) factory.create(shape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor[i]);
		}
	 	return outputImg;
	}
}
