/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the BioImage.io nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
package io.bioimage.modelrunner.numpy;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import io.bioimage.modelrunner.utils.IndexingUtils;

import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.cell.CellImgFactory;
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
 * Class to convert numpy arrays stored in npy files into ImgLib2 images
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class DecodeNumpy {

	/**
	 * Size of the chunks used to read the bytes 
	 */
    private static final int BUFFER_SIZE = 1024 * 1024;
    /**
     * Numpy file extension
     */
    private static final String NUMPY_EXTENSION = ".npy";
    /**
     * Prefix used to identify every numpy array
     */
    private static final byte[] MAGIC_PREFIX = {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'};
    /**
     * Map containing the relation between the datatypes used by numpy and their 
     * explicit name
     */
    private static final Map<String, Integer> DATA_TYPES_MAP = new HashMap<>();

    static
    {
        DATA_TYPES_MAP.put("byte", 1);
        DATA_TYPES_MAP.put("int16", 2);
        DATA_TYPES_MAP.put("int32", 4);
        DATA_TYPES_MAP.put("int64", 8);
        DATA_TYPES_MAP.put("boolean", 1);
        DATA_TYPES_MAP.put("uint8", 4);
        DATA_TYPES_MAP.put("uint16", 4);
        DATA_TYPES_MAP.put("uint32", 8);
        DATA_TYPES_MAP.put("float16", 4);
        DATA_TYPES_MAP.put("float32", 4);
        DATA_TYPES_MAP.put("float64", 8);
    }

    /**
     * PAttern that matches the metadata description of a numpy file
     */
    private static final Pattern HEADER_PATTERN =
            Pattern.compile("\\{'descr': '(.+)', 'fortran_order': False, 'shape': \\((.*)\\),");
	
    /**
     * Main method to test the ImgLib2 creation
     * @param args
     * 	no args are needed
     * @throws FileNotFoundException if the numpy array file is not found
     * @throws IOException if there is any error opening the files
     */
    public static void main(String[] args) throws FileNotFoundException, IOException {
    	String npy = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\models\\HPA Bestfitting InceptionV3_13102022_173532\\test_input.npy";
    	RandomAccessibleInterval<?> aa = retrieveImgLib2FromNpy(npy);
    }
    
    /**
     * Method that retreives an ImgLib2 image from a numpy array stored in the file specified by 
     * the argument
     * @param <T>
     * 	possible data types that the ImgLib2 image can have
     * @param path
     * 	path to the file where the .npy file containing the numpy array is stored
     * @return an ImgLib2 image with the same datatype, shape and data that the numpy array
     * @throws FileNotFoundException if the numpy file is not found
     * @throws IOException if there is any error opening the numpy file
     */
    public static < T extends RealType< T > & NativeType< T > > 
								RandomAccessibleInterval<T> retrieveImgLib2FromNpy(String path) throws FileNotFoundException, IOException{
    	File npyFile = new File(path);
    	if (!npyFile.isFile() || !path.endsWith(NUMPY_EXTENSION)) {
    		throw new IllegalArgumentException("Path provided does not correspond to a Numpy file: " + path);
    	}
    	try (InputStream targetStream = new FileInputStream(npyFile)) {
    		return decodeNumpy(targetStream);
    	}
    }
    
    /**
     * MEthod to decode the bytes corresponding to a numpy array stored in the numpy file
     * @param <T>
     * 	possible data types that the ImgLib2 image can have
     * @param is
     * 	{@link InputStream} that results after reading the numpy file. Contains the byte info of the
     * 	numpy array
     * @return an ImgLib2 image with the same datatype, shape and data that the numpy array
     * @throws IOException if there is any error reading the {@link InputStream}
     */
    private static < T extends RealType< T > & NativeType< T > > 
    				RandomAccessibleInterval<T> decodeNumpy(InputStream is) throws IOException {
        DataInputStream dis;
        if (is instanceof DataInputStream) {
            dis = (DataInputStream) is;
        } else {
            dis = new DataInputStream(is);
        }

        byte[] buf = new byte[MAGIC_PREFIX.length];
        dis.readFully(buf);
        if (!Arrays.equals(buf, MAGIC_PREFIX)) {
            throw new IllegalArgumentException("Malformed  or unsopported Numpy array");
        }
        byte major = dis.readByte();
        byte minor = dis.readByte();
        if (major < 1 || major > 3 || minor != 0) {
            throw new IllegalArgumentException("Unknown numpy version: " + major + '.' + minor);
        }
        int len = major == 1 ? 2 : 4;
        dis.readFully(buf, 0, len);
        ByteBuffer bb = ByteBuffer.wrap(buf, 0, len);
        bb.order(ByteOrder.LITTLE_ENDIAN);
        if (major == 1) {
            len = bb.getShort();
        } else {
            len = bb.getInt();
        }
        buf = new byte[len];
        dis.readFully(buf);
        String header = new String(buf, StandardCharsets.UTF_8).trim();
        Matcher m = HEADER_PATTERN.matcher(header);
        if (!m.find()) {
            throw new IllegalArgumentException("Invalid numpy header: " + header);
        }
        String typeStr = m.group(1);
        String shapeStr = m.group(2);
        long[] shape = new long[0];
        if (!shapeStr.isEmpty()) {
            String[] tokens = shapeStr.split(", ?");
            shape = Arrays.stream(tokens).mapToLong(Long::parseLong).toArray();
        }
        String dtype = getDataType(typeStr);
        long numBytes = DATA_TYPES_MAP.get(dtype);
    	long count;
    	if (shape.length == 0)
    		count = 1;
		else
			count = Arrays.stream(shape).reduce(Math::multiplyExact).getAsLong();
        //len = Math.toIntExact(shape.length * numBytes);
        len = Math.toIntExact(count * numBytes);
        ByteBuffer data = ByteBuffer.allocate(len);
        char order = typeStr.charAt(0);
        ByteOrder byteOrder = null;
        if (order == '>') {
        	byteOrder = ByteOrder.BIG_ENDIAN;
        } else if (order == '<') {
        	byteOrder = ByteOrder.LITTLE_ENDIAN;
        } else {
        	new IllegalArgumentException("Not supported ByteOrder for the provided .npy array.");
        }
        data.order(byteOrder);
        readData(dis, data, len);

        return build(data, byteOrder, dtype, shape);
    }
    
    /**
     * Get a String representing a datatype explicitly from the String that numpy uses to
     * name datatypes
     * @param npDtype
     * 	datatype defined per Numpy
     * @return a String defining the datatype in a explicit manner
     * @throws IllegalArgumentException if the String provided is not a numpy datatype
     */
    public static String getDataType(String npDtype) throws IllegalArgumentException {
    	if (npDtype.startsWith(">") || npDtype.startsWith("<"))
    		npDtype = npDtype.substring(1);
    	if (npDtype.equals("i1") || npDtype.equals("b") || npDtype.equals("c"))
    		return "byte";
    	else if (npDtype.equals("i2") || npDtype.equals("h"))
    		return "int16";
    	else if (npDtype.equals("i4") || npDtype.equals("i"))
    		return "int32";
    	else if (npDtype.equals("i8") || npDtype.equals("l")
    			|| npDtype.equals("q"))
    		return "int64";
    	else if (npDtype.equals("b1"))
    		return "boolean";
    	else if (npDtype.equals("u1") || npDtype.equals("B"))
    		return "uint8";
    	else if (npDtype.equals("u2") || npDtype.equals("H"))
    		return "uint16";
    	else if (npDtype.equals("u4") || npDtype.equals("I"))
    		return "uint32";
    	else if (npDtype.equals("f2") || npDtype.equals("e"))
    		return "float16";
    	else if (npDtype.equals("f") || npDtype.equals("f4"))
    		return "float32";
    	else if (npDtype.equals("f8") || npDtype.equals("d"))
    		return "float64";
    	else if (npDtype.equals("u8") || npDtype.equals("L")
    			|| npDtype.equals("Q"))
    		throw new IllegalArgumentException("Numpy dtype 'uint64' cannot "
    				+ " be supported in Java.");
    	else if (npDtype.equals("c8"))
    		throw new IllegalArgumentException("Numpy dtype 'complex64' is not "
    				+ "supported at the moment.");
    	else
    		throw new IllegalArgumentException("Numpy dtype '" + npDtype + "' is not "
    				+ "supported at the moment.");
    }

    /**
     * Read the data from the input stream into a byte buffer
     * @param dis
     * 	the {@link DataInputStream} from where the data is read from
     * @param data
     * 	{@link ByteBuffer} where the info is copied to
     * @param len
     * 	remaining number of bytes in the {@link DataInputStream}
     * @throws IOException if there is any error reading the {@link DataInputStream}
     */
    private static void readData(DataInputStream dis, ByteBuffer data, int len) throws IOException {
        if (len > 0) {
            byte[] buf = new byte[BUFFER_SIZE];
            while (len > BUFFER_SIZE) {
                dis.readFully(buf);
                data.put(buf);
                len -= BUFFER_SIZE;
            }

            dis.readFully(buf, 0, len);
            data.put(buf, 0, len);
            data.rewind();
        }
    }

    /**
     * Creates a {@link Img} from a given {@link ByteBuffer} and an array with its dimensions order.
     * 
     * @param <T>
     * 		  possible ImgLib2 dataypes that the img can be
     * @param buf
     *        The buffer data is read from.
	 * @param byteOrder
	 *        Endianness of the buffer data.
	 * @param dtype
	 *        NumPy dtype of the data.
	 * @param shape
	 *        NumPy shape of the data.
     * @return The Img built from the tensor.
     * @throws IllegalArgumentException
     *         If the tensor type is not supported.
     */
    @SuppressWarnings("unchecked")
    public static <T extends Type<T>> Img<T> build(ByteBuffer buf, ByteOrder byteOrder, String dtype, long[] shape) throws IllegalArgumentException
    {
        // Create an Img of the same type of the tensor
    	byte[] data = new byte[buf.remaining()];
    	buf.get(data);
    	if (dtype.equals("byte")) {
            return (Img<T>) buildByteFromByte(data, byteOrder, shape);
    	} else if (dtype.equals("ubyte")) {
            return (Img<T>) buildUByteFromByte(data, byteOrder, shape);
    	} else if (dtype.equals("int16")) {
            return (Img<T>) buildInt16FromByte(data, byteOrder, shape);
    	} else if (dtype.equals("uint16")) {
            return (Img<T>) buildUInt16FromByte(data, byteOrder, shape);
    	} else if (dtype.equals("int32")) {
            return (Img<T>) buildInt32FromByte(data, byteOrder, shape);
    	} else if (dtype.equals("uint32")) {
            return (Img<T>) buildUInt32FromByte(data, byteOrder, shape);
    	} else if (dtype.equals("int64")) {
            return (Img<T>) buildInt64FromByte(data, byteOrder, shape);
    	} else if (dtype.equals("float32")) {
            return (Img<T>) buildFloat32FromByte(data, byteOrder, shape);
    	} else if (dtype.equals("float64")) {
            return (Img<T>) buildFloat64FromByte(data, byteOrder, shape);
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + dtype);
    	}
    }

    /** TODO check BigEndian LittleEndian
     * Builds a {@link Img} from a unsigned byte-typed {@code Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The Img built from the tensor of type {@link DataType#UBYTE}.
     */
    private static Img<ByteType> buildByteFromByte(byte[] tensor, ByteOrder byteOrder, long[] tensorShape)
    {
    	final ImgFactory< ByteType > factory = new CellImgFactory<>( new ByteType(), 5 );
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(tensorShape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	byte val = tensor[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
	}

    /** 
     * Builds a {@link Img} from a unsigned byte-typed {@code Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The Img built from the tensor of type {@link DataType#UBYTE}.
     */
    private static Img<UnsignedByteType> buildUByteFromByte(byte[] tensor, ByteOrder byteOrder, long[] tensorShape)
    {
    	final ImgFactory< UnsignedByteType > factory = new CellImgFactory<>( new UnsignedByteType(), 5 );
        final Img< UnsignedByteType > outputImg = (Img<UnsignedByteType>) factory.create(tensorShape);
    	Cursor<UnsignedByteType> tensorCursor= outputImg.cursor();
        int[] flatArr = ByteArrayUtils.convertIntoUInt8(tensor, byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	int val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
	}

    /**
     * Builds a {@link Img} from a unsigned integer-typed {@code Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The sequence built from the tensor of type {@link DataType#INT}.
     */
    private static Img<ShortType> buildInt16FromByte(byte[] tensor, ByteOrder byteOrder, long[] tensorShape)
    {
    	final ImgFactory< ShortType > factory = new CellImgFactory<>( new ShortType(), 5 );
        final Img< ShortType > outputImg = (Img<ShortType>) factory.create(tensorShape);
    	Cursor<ShortType> tensorCursor= outputImg.cursor();
        short[] flatArr = ByteArrayUtils.convertIntoSignedShort16(tensor, byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	short val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
     * Builds a {@link Img} from a unsigned integer-typed {@code Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The sequence built from the tensor of type {@link DataType#INT}.
     */
    private static Img<UnsignedShortType> buildUInt16FromByte(byte[] tensor, ByteOrder byteOrder, long[] tensorShape)
    {
    	final ImgFactory< UnsignedShortType > factory = new CellImgFactory<>( new UnsignedShortType(), 5 );
        final Img< UnsignedShortType > outputImg = (Img<UnsignedShortType>) factory.create(tensorShape);
    	Cursor<UnsignedShortType> tensorCursor= outputImg.cursor();
        int[] flatArr = ByteArrayUtils.convertIntoUnsignedInt16(tensor, byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	int val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
     * Builds a {@link Img} from a unsigned integer-typed {@code Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The sequence built from the tensor of type {@link DataType#INT}.
     */
    private static Img<IntType> buildInt32FromByte(byte[] tensor, ByteOrder byteOrder, long[] tensorShape)
    {
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
        final Img< IntType > outputImg = (Img<IntType>) factory.create(tensorShape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
        int[] flatArr = ByteArrayUtils.convertIntoSignedInt32(tensor, byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	int val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
     * Builds a {@link Img} from a unsigned integer-typed {@code Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The sequence built from the tensor of type {@link DataType#INT}.
     */
    private static Img<UnsignedIntType> buildUInt32FromByte(byte[] tensor, ByteOrder byteOrder, long[] tensorShape)
    {
    	final ImgFactory< UnsignedIntType > factory = new CellImgFactory<>( new UnsignedIntType(), 5 );
        final Img< UnsignedIntType > outputImg = (Img<UnsignedIntType>) factory.create(tensorShape);
    	Cursor<UnsignedIntType> tensorCursor= outputImg.cursor();
        long[] flatArr = ByteArrayUtils.convertIntoUnsignedInt32(tensor, byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	long val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
     * Builds a {@link Img} from a unsigned float-typed {@code Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The Img built from the tensor of type {@link DataType#FLOAT}.
     */
    private static Img<FloatType> buildFloat32FromByte(byte[] tensor, ByteOrder byteOrder, long[] tensorShape)
    {
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(tensorShape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
        float[] flatArr = ByteArrayUtils.convertIntoSignedFloat32(tensor, byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	float val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
     * Builds a {@link Img} from a unsigned double-typed {@code Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The Img built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<DoubleType> buildFloat64FromByte(byte[] tensor, ByteOrder byteOrder, long[] tensorShape)
    {
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(tensorShape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
		double[] flatArr = ByteArrayUtils.convertIntoSignedFloat64(tensor, byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	double val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
     * Builds a {@link Img} from a unsigned double-typed {@code Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The Img built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<LongType> buildInt64FromByte(byte[] tensor, ByteOrder byteOrder, long[] tensorShape)
    {
    	final ImgFactory< LongType > factory = new CellImgFactory<>( new LongType(), 5 );
        final Img< LongType > outputImg = (Img<LongType>) factory.create(tensorShape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
		long[] flatArr = ByteArrayUtils.convertIntoSignedInt64(tensor, byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	long val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

}
