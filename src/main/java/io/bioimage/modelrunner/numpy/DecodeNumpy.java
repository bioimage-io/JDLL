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
import net.imglib2.img.array.ArrayImgFactory;
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

/**
 * TODO
 * TODO
 * TODO ADD imglib2 to numpy file
 * 
 * 
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
        DATA_TYPES_MAP.put("boolean", 1);
        DATA_TYPES_MAP.put("int8", 1);
        DATA_TYPES_MAP.put("uint8", 1);
        DATA_TYPES_MAP.put("int16", 2);
        DATA_TYPES_MAP.put("uint16", 1);
        DATA_TYPES_MAP.put("int32", 4);
        DATA_TYPES_MAP.put("uint32", 4);
        DATA_TYPES_MAP.put("int64", 8);
        DATA_TYPES_MAP.put("float16", 2);
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
        char order = typeStr.charAt(0);
        ByteOrder byteOrder = null;
        if (order == '>') {
        	byteOrder = ByteOrder.BIG_ENDIAN;
        } else if (order == '<') {
        	byteOrder = ByteOrder.LITTLE_ENDIAN;
        } else if (order == '|') {
        	byteOrder = ByteOrder.LITTLE_ENDIAN;
        	new IOException("Numpy .npy file did not specify the byte order of the array."
        			+ " It was automatically opened as little endian but this does not guarantee"
        			+ " the that the file is open correctly. Caution is advised.").printStackTrace();
    	} else {
        	new IllegalArgumentException("Not supported ByteOrder for the provided .npy array.");
        }
        String dtype = getDataType(typeStr.substring(1));
        long numBytes = DATA_TYPES_MAP.get(dtype);
    	long count;
    	if (shape.length == 0)
    		count = 1;
		else
			count = Arrays.stream(shape).reduce(Math::multiplyExact).getAsLong();
        //len = Math.toIntExact(shape.length * numBytes);
        len = Math.toIntExact(count * numBytes);
        ByteBuffer data = ByteBuffer.allocate(len);
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
    	if (npDtype.equals("i1") || npDtype.equals("b") || npDtype.equals("c"))
    		return "int8";
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
    public static <T extends NativeType<T>> Img<T> build(ByteBuffer buf, ByteOrder byteOrder, String dtype, long[] shape) throws IllegalArgumentException
    {
    	if (dtype.equals("int8")) {
    		return (Img<T>) buildInt8(buf, byteOrder, shape);
    	} else if (dtype.equals("uint8")) {
    		return (Img<T>) buildUInt8(buf, byteOrder, shape);
    	} else if (dtype.equals("int16")) {
    		return (Img<T>) buildInt16(buf, byteOrder, shape);
    	} else if (dtype.equals("uint16")) {
    		return (Img<T>) buildUInt16(buf, byteOrder, shape);
    	} else if (dtype.equals("int32")) {
    		return (Img<T>) buildInt32(buf, byteOrder, shape);
    	} else if (dtype.equals("uint32")) {
    		return (Img<T>) buildUInt32(buf, byteOrder, shape);
    	} else if (dtype.equals("int64")) {
    		return (Img<T>) buildInt64(buf, byteOrder, shape);
    	} else if (dtype.equals("float32")) {
    		return (Img<T>) buildFloat32(buf, byteOrder, shape);
    	} else if (dtype.equals("float64")) {
    		return (Img<T>) buildFloat64(buf, byteOrder, shape);
    	} else if (dtype.equals("bool")) {
    		return (Img<T>) buildBoolean(buf, byteOrder, shape);
    	} else {
            throw new IllegalArgumentException("Unsupported data type of numpy array: " + dtype);
    	}
    }
    
    private static Img<ByteType> buildInt8(ByteBuffer buf, ByteOrder byteOrder, long[] shape) {
    	buf.order(byteOrder);
		final ArrayImgFactory< ByteType > factory = new ArrayImgFactory<>( new ByteType() );
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(shape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, shape);
        	tensorCursor.get().set(buf.array()[flatPos]);
		}
	 	return outputImg;
    }
    
    private static Img<UnsignedByteType> buildUInt8(ByteBuffer buf, ByteOrder byteOrder, long[] shape) {
    	buf.order(byteOrder);
		final ArrayImgFactory< UnsignedByteType > factory = new ArrayImgFactory<>( new UnsignedByteType() );
        final Img< UnsignedByteType > outputImg = (Img<UnsignedByteType>) factory.create(shape);
    	Cursor<UnsignedByteType> tensorCursor= outputImg.cursor();
    	int[] flatArr = ByteArrayUtils.toUInt8(buf.array(), byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, shape);
        	tensorCursor.get().set(flatArr[flatPos]);
		}
	 	return outputImg;
    }
    
    private static Img<ShortType> buildInt16(ByteBuffer buf, ByteOrder byteOrder, long[] shape) {
    	buf.order(byteOrder);
		final ArrayImgFactory< ShortType > factory = new ArrayImgFactory<>( new ShortType() );
        final Img< ShortType > outputImg = (Img<ShortType>) factory.create(shape);
    	Cursor<ShortType> tensorCursor= outputImg.cursor();
    	short[] flatArr = ByteArrayUtils.toInt16(buf.array(), byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, shape);
        	tensorCursor.get().set(flatArr[flatPos]);
		}
	 	return outputImg;
    	
    }
    
    private static Img<UnsignedShortType> buildUInt16(ByteBuffer buf, ByteOrder byteOrder, long[] shape) {
    	buf.order(byteOrder);
		final ArrayImgFactory< UnsignedShortType > factory = new ArrayImgFactory<>( new UnsignedShortType() );
        final Img< UnsignedShortType > outputImg = (Img<UnsignedShortType>) factory.create(shape);
    	Cursor<UnsignedShortType> tensorCursor= outputImg.cursor();
    	int[] flatArr = ByteArrayUtils.toUInt16(buf.array(), byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, shape);
        	tensorCursor.get().set(flatArr[flatPos]);
		}
	 	return outputImg;
    }
    
    private static Img<IntType> buildInt32(ByteBuffer buf, ByteOrder byteOrder, long[] shape) {
    	buf.order(byteOrder);
		final ArrayImgFactory< IntType > factory = new ArrayImgFactory<>( new IntType() );
        final Img< IntType > outputImg = (Img<IntType>) factory.create(shape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
    	int[] flatArr = ByteArrayUtils.toInt32(buf.array(), byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, shape);
        	tensorCursor.get().set(flatArr[flatPos]);
		}
	 	return outputImg;
    }
    
    private static Img<UnsignedIntType> buildUInt32(ByteBuffer buf, ByteOrder byteOrder, long[] shape) {
    	buf.order(byteOrder);
		final ArrayImgFactory< UnsignedIntType > factory = new ArrayImgFactory<>( new UnsignedIntType() );
        final Img< UnsignedIntType > outputImg = (Img<UnsignedIntType>) factory.create(shape);
    	Cursor<UnsignedIntType> tensorCursor= outputImg.cursor();
    	long[] flatArr = ByteArrayUtils.toUInt32(buf.array(), byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, shape);
        	tensorCursor.get().set(flatArr[flatPos]);
		}
	 	return outputImg;
    }
    
    private static Img<LongType> buildInt64(ByteBuffer buf, ByteOrder byteOrder, long[] shape) {
    	buf.order(byteOrder);
		final ArrayImgFactory< LongType > factory = new ArrayImgFactory<>( new LongType() );
        final Img< LongType > outputImg = (Img<LongType>) factory.create(shape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
    	long[] flatArr = ByteArrayUtils.toInt64(buf.array(), byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, shape);
        	tensorCursor.get().set(flatArr[flatPos]);
		}
	 	return outputImg;
    }
    
    private static Img<FloatType> buildFloat32(ByteBuffer buf, ByteOrder byteOrder, long[] shape) {
    	buf.order(byteOrder);
		final ArrayImgFactory< FloatType > factory = new ArrayImgFactory<>( new FloatType() );
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(shape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
    	float[] flatArr = ByteArrayUtils.toFloat32(buf.array(), byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, shape);
        	float val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }
    
    private static Img<DoubleType> buildFloat64(ByteBuffer buf, ByteOrder byteOrder, long[] shape) {
    	buf.order(byteOrder);
		final ArrayImgFactory< DoubleType > factory = new ArrayImgFactory<>( new DoubleType() );
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(shape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
    	double[] flatArr = ByteArrayUtils.toFloat64(buf.array(), byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, shape);
        	tensorCursor.get().set(flatArr[flatPos]);
		}
	 	return outputImg;
    }
    
    private static Img<ByteType> buildBoolean(ByteBuffer buf, ByteOrder byteOrder, long[] shape) {
    	buf.order(byteOrder);
		final ArrayImgFactory< ByteType > factory = new ArrayImgFactory<>( new ByteType() );
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(shape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
    	boolean[] flatArr = ByteArrayUtils.toBoolean(buf.array(), byteOrder);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, shape);
        	tensorCursor.get().set((byte)(flatArr[flatPos]?1:0));
		}
	 	return outputImg;
    }
}
