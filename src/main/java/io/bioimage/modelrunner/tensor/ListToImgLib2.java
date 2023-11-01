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

import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.stream.IntStream;

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
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * A {@link Img} builder from {@link ByteBuffer} objects
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
    		buildTensor(List array, List<Integer> shape, String axes, String dtype, String name) 
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
    public static < T extends RealType< T > & NativeType< T > > Img<T> build(List array, List<Integer> shape, String dtype) throws IllegalArgumentException
    {
    	if (shape.size() == 0 || array.size() == 0)
    		return null;
    	
        Img<T> data;
		switch (dtype)
        {
	    	case "int8":
	            data = (Img<T>) buildInt8(array, shape);
	            break;
	    	case "uint8":
	            data = (Img<T>) buildUint8(array, shape);
	            break;
	    	case "int16":
	            data = (Img<T>) buildInt16(array, shape);
	            break;
	    	case "uint16":
	            data = (Img<T>) buildUint16(array, shape);
	            break;
            case "int32":
            	data = (Img<T>) buildInt32(array, shape);
                break;
            case "uint32":
            	data = (Img<T>) buildUint32(array, shape);
                break;
            case "int64":
            	data = (Img<T>) buildInt64(array, shape);
                break;
            case "float32":
            	data = (Img<T>) buildFloat32(array, shape);
                break;
            case "float64":
            	data = (Img<T>) buildFloat64(array, shape);
                break;
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + dtype);
        }
		return data;
    }
    
    @SuppressWarnings("unchecked")
	private static < T extends RealType< T > & NativeType< T > > Img<T> createEmptyArray(List<Integer> tensorShape, String dtype) {
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();

        final Img<T> data;
        final ArrayImgFactory<?> factory;
		switch (dtype)
        {
	    	case "int8":
	        	factory = new ArrayImgFactory<>( new ByteType() );
	        	data = (Img<T>) factory.create(shape);
	        	break;
	    	case "uint8":
	        	factory = new ArrayImgFactory<>( new UnsignedByteType() );
	        	data = (Img<T>) factory.create(shape);
	            break;
	    	case "int16":
	        	factory = new ArrayImgFactory<>( new ShortType() );
	        	data = (Img<T>) factory.create(shape);
	            break;
	    	case "uint16":
	        	factory = new ArrayImgFactory<>( new UnsignedShortType() );
	        	data = (Img<T>) factory.create(shape);
	            break;
            case "int32":
	        	factory = new ArrayImgFactory<>( new IntType() );
	        	data = (Img<T>) factory.create(shape);
                break;
            case "uint32":
	        	factory = new ArrayImgFactory<>( new UnsignedIntType() );
	        	data = (Img<T>) factory.create(shape);
                break;
            case "int64":
	        	factory = new ArrayImgFactory<>( new LongType() );
	        	data = (Img<T>) factory.create(shape);
                break;
            case "float32":
	        	factory = new ArrayImgFactory<>( new FloatType() );
	        	data = (Img<T>) factory.create(shape);
                break;
            case "float64":
	        	factory = new ArrayImgFactory<>( new DoubleType() );
	        	data = (Img<T>) factory.create(shape);
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
    private static Img<ByteType> buildInt8(List array, List<Integer> tensorShape)
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
     * Builds a ByteType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<ByteType> buildInt8FromByte(List<Byte> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< ByteType > factory = new ArrayImgFactory<>( new ByteType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(shape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i));
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
    private static Img<ByteType> buildInt8FromInteger(List<Integer> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< ByteType > factory = new ArrayImgFactory<>( new ByteType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(shape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i).byteValue());
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
    private static Img<UnsignedByteType> buildUint8(List array, List<Integer> tensorShape)
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
     * Builds a ByteType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<UnsignedByteType> buildUint8FromByte(List<Byte> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< UnsignedByteType > factory = new ArrayImgFactory<>( new UnsignedByteType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< UnsignedByteType > outputImg = (Img<UnsignedByteType>) factory.create(shape);
    	Cursor<UnsignedByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i));
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
    private static Img<UnsignedByteType> buildUint8FromInteger(List<Integer> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< UnsignedByteType > factory = new ArrayImgFactory<>( new UnsignedByteType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< UnsignedByteType > outputImg = (Img<UnsignedByteType>) factory.create(shape);
    	Cursor<UnsignedByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i).byteValue());
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
    private static Img<ShortType> buildInt16(List array, List<Integer> tensorShape)
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
     * Builds a IntType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<ShortType> buildInt16FromShort(List<Short> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< ShortType > factory = new ArrayImgFactory<>( new ShortType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< ShortType > outputImg = (Img<ShortType>) factory.create(shape);
    	Cursor<ShortType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i));
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
    private static Img<ShortType> buildInt16FromInteger(List<Integer> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< ShortType > factory = new ArrayImgFactory<>( new ShortType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< ShortType > outputImg = (Img<ShortType>) factory.create(shape);
    	Cursor<ShortType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i).shortValue());
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
    private static Img<UnsignedShortType> buildUint16(List array, List<Integer> tensorShape)
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
     * Builds a IntType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<UnsignedShortType> buildUint16FromShort(List<Short> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< UnsignedShortType > factory = new ArrayImgFactory<>( new UnsignedShortType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< UnsignedShortType > outputImg = (Img<UnsignedShortType>) factory.create(shape);
    	Cursor<UnsignedShortType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i));
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
    private static Img<UnsignedShortType> buildUint16FromInteger(List<Integer> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< UnsignedShortType > factory = new ArrayImgFactory<>( new UnsignedShortType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< UnsignedShortType > outputImg = (Img<UnsignedShortType>) factory.create(shape);
    	Cursor<UnsignedShortType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i));
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
    private static Img<IntType> buildInt32(List array, List<Integer> tensorShape)
    {
    	if (!(array.get(0) instanceof Integer)) {
    		throw new IllegalArgumentException("Unable to build ImgLib2 array of data type "
    				+ "'int32' using Java array of class: " + array.get(0).getClass());
    	} else {
    		return buildInt32FromInteger((List<Integer>) array, tensorShape);
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
    private static Img<IntType> buildInt32FromInteger(List<Integer> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< IntType > factory = new ArrayImgFactory<>( new IntType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< IntType > outputImg = (Img<IntType>) factory.create(shape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i));
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
    private static Img<UnsignedIntType> buildUint32(List array, List<Integer> tensorShape)
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
     * Builds a IntType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<UnsignedIntType> buildUint32FromInteger(List<Integer> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< UnsignedIntType > factory = new ArrayImgFactory<>( new UnsignedIntType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< UnsignedIntType > outputImg = (Img<UnsignedIntType>) factory.create(shape);
    	Cursor<UnsignedIntType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i));
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
    private static Img<UnsignedIntType> buildUint32FromLong(List<Long> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< UnsignedIntType > factory = new ArrayImgFactory<>( new UnsignedIntType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< UnsignedIntType > outputImg = (Img<UnsignedIntType>) factory.create(shape);
    	Cursor<UnsignedIntType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i));
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
    private static Img<LongType> buildInt64(List array, List<Integer> tensorShape)
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
     * Builds a LongType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<LongType> buildInt64FromLong(List<Long> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< LongType > factory = new ArrayImgFactory<>( new LongType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< LongType > outputImg = (Img<LongType>) factory.create(shape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i));
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
    private static Img<LongType> buildInt64FromInteger(List<Integer> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< LongType > factory = new ArrayImgFactory<>( new LongType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< LongType > outputImg = (Img<LongType>) factory.create(shape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i));
		}
	 	return outputImg;
	}

    /**
     * Builds a FloatType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<FloatType> buildFloat32(List array, List<Integer> tensorShape)
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
     * Builds a FloatType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<FloatType> buildFloat32FromFloat(List<Float> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< FloatType > factory = new ArrayImgFactory<>( new FloatType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(shape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i));
		}
	 	return outputImg;
	}

    /**
     * Builds a FloatType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<FloatType> buildFloat32FromBigDecimal(List<BigDecimal> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< FloatType > factory = new ArrayImgFactory<>( new FloatType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(shape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i).floatValue());
		}
	 	return outputImg;
	}

    /**
     * Builds a DoubleType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<DoubleType> buildFloat64(List array, List<Integer> tensorShape)
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
     * Builds a DoubleType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<DoubleType> buildFloat64FromDouble(List<Double> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< DoubleType > factory = new ArrayImgFactory<>( new DoubleType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(shape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i));
		}
	 	return outputImg;
	}

    /**
     * Builds a DoubleType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<DoubleType> buildFloat64FromFloat(List<Float> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< DoubleType > factory = new ArrayImgFactory<>( new DoubleType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(shape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i));
		}
	 	return outputImg;
	}

    /**
     * Builds a DoubleType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<DoubleType> buildFloat64FromBigDecimal(List<BigDecimal> tensor, List<Integer> tensorShape)
    {
    	final ArrayImgFactory< DoubleType > factory = new ArrayImgFactory<>( new DoubleType() );
    	long[] shape = IntStream.range(0, tensorShape.size()).mapToLong(i -> tensorShape.get(i)).toArray();
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(shape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			int i = IndexingUtils.multidimensionalIntoFlatIndex(tensorCursor.positionAsLongArray(),
					shape);
        	tensorCursor.get().set(tensor.get(i).doubleValue());
		}
	 	return outputImg;
	}
}