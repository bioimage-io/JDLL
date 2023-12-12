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
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import io.bioimage.modelrunner.tensor.ImgLib2ToArray;
import io.bioimage.modelrunner.tensor.Utils;
import io.bioimage.modelrunner.utils.IndexingUtils;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
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
 * 
 * 
 * Class to convert numpy arrays stored in npy files into ImgLib2 images and viceversa
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
    public static final Map<String, Integer> DATA_TYPES_MAP = new HashMap<>();

    static
    {
        DATA_TYPES_MAP.put("boolean", 1);
        DATA_TYPES_MAP.put("int8", 1);
        DATA_TYPES_MAP.put("uint8", 1);
        DATA_TYPES_MAP.put("int16", 2);
        DATA_TYPES_MAP.put("uint16", 2);
        DATA_TYPES_MAP.put("int32", 4);
        DATA_TYPES_MAP.put("uint32", 4);
        DATA_TYPES_MAP.put("int64", 8);
        DATA_TYPES_MAP.put("float16", 2);
        DATA_TYPES_MAP.put("float32", 4);
        DATA_TYPES_MAP.put("float64", 8);
    }

    /**
     * Key used to refer to the values of the array
     */
    public final static String DATA_KEY = "data";
    /**
     * Key referred to the shape of the array (array dimensions)
     */
    public final static String SHAPE_KEY = "shape";
    /**
     * Key referred to the shape of the array data type
     */
    public final static String DTYPE_KEY = "dtype";
    /**
     * Key referred to the order when flattening the array, c-order or fortran-order
     */
    public final static String IS_FORTRAN_ORDER_KEY = "is_fortran_order";
    /**
     * Key referred to the byte order in the array
     */
    public final static String BYTE_ORDER_KEY = "byte_order";

    /**
     * PAttern that matches the metadata description of a numpy file
     */
    public static final Pattern HEADER_PATTERN =
            Pattern.compile("\\{'descr': '(.+)', 'fortran_order': (True|False), 'shape': \\((.*)\\),");
	
    /**
     * Main method to test the ImgLib2 creation
     * @param <T>
     * 	possible ImgLib2 data types of the provided {@link RandomAccessibleInterval}
     * @param args
     * 	no args are needed
     * @throws FileNotFoundException if the numpy array file is not found
     * @throws IOException if there is any error opening the files
     */
    public static < T extends RealType< T > & NativeType< T > > void main(String[] args) throws FileNotFoundException, IOException {
    	//String npy = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\models\\Arabidopsis Leaf Segmentation_30102023_193340\\test_input.npy";
    	//RandomAccessibleInterval<?> aa = retrieveImgLib2FromNpy(npy);
    	String npy = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\test_input.npy";
    	RandomAccessibleInterval<T> rai = Cast.unchecked(ArrayImgs.doubles(new long[] {1, 512, 512}));
    	writeRaiToNpyFile(npy, rai);
    	RandomAccessibleInterval<?> bb = retrieveImgLib2FromNpy(npy);
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
    		return decodeNumpyFromByteArrayStream(targetStream);
    	}
    }
    
    /**
     * MEthod to decode the bytes corresponding to a numpy array stored in the numpy file
     * @param is
     * 	{@link InputStream} that results after reading the numpy file. Contains the byte info of the
     * 	numpy array
     * @return an ImgLib2 image with the same datatype, shape and data that the numpy array
     * @throws IOException if there is any error reading the {@link InputStream}
     */
    public static HashMap<String, Object> decodeNumpyFromByteArrayStreamToRawMap(InputStream is) throws IOException {
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
        String header = new String(buf, StandardCharsets.UTF_8);
        Matcher m = HEADER_PATTERN.matcher(header);
        if (!m.find()) {
            throw new IllegalArgumentException("Invalid numpy header: " + header);
        }
        String typeStr = m.group(1);
        String fortranOrder = m.group(2).trim();
        String shapeStr = m.group(3);
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

        HashMap<String, Object> map = new HashMap<String, Object>();
        map.put(SHAPE_KEY, shape);
        map.put(BYTE_ORDER_KEY, byteOrder);
        map.put(DTYPE_KEY, dtype);
        map.put(IS_FORTRAN_ORDER_KEY, fortranOrder.equals("True"));
        map.put(DATA_KEY, data);

        return map;
    }
    
    /**
     * MEthod to decode the bytes corresponding to a numpy array stored in the numpy file
     * and convert them into a {@link RandomAccessibleInterval}
     * @param <T>
     * 	possible data types that the ImgLib2 image can have
     * @param is
     * 	{@link InputStream} that results after reading the numpy file. Contains the byte info of the
     * 	numpy array
     * @return an ImgLib2 image with the same datatype, shape and data that the numpy array
     * @throws IOException if there is any error reading the {@link InputStream}
     */
    public static < T extends RealType< T > & NativeType< T > > 
    				RandomAccessibleInterval<T> decodeNumpyFromByteArrayStream(InputStream is) throws IOException {
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
        String header = new String(buf, StandardCharsets.UTF_8);
        Matcher m = HEADER_PATTERN.matcher(header);
        if (!m.find()) {
            throw new IllegalArgumentException("Invalid numpy header: " + header);
        }
        String typeStr = m.group(1);
        String fortranOrder = m.group(2).trim();
        String shapeStr = m.group(3);
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

        return build(data, byteOrder, dtype, shape, fortranOrder.equals("True"));
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
     * Get a String representing a datatype explicitly from the String that numpy uses to
     * name datatypes
	 * @param <T>
     * 	possible ImgLib2 data types 
     * @param type
     * 	ImgLib2 possible data type
     * @return a String defining the datatype in a explicit manner
     * @throws IllegalArgumentException if the String provided is not a numpy datatype
     */
    public static  <T extends RealType<T> & NativeType<T>> 
    String getDataType(T type) throws IllegalArgumentException {
    	if (type instanceof ByteType)
    		return "i1";
    	else if (type instanceof ShortType)
    		return "i2";
    	else if (type instanceof IntType)
    		return "i4";
    	else if (type instanceof LongType)
    		return "i8";
    	else if (type instanceof UnsignedByteType)
    		return "u1";
    	else if (type instanceof UnsignedShortType)
    		return "u2";
    	else if (type instanceof UnsignedIntType)
    		return "u4";
    	else if (type instanceof FloatType)
    		return "f4";
    	else if (type instanceof DoubleType)
    		return "f8";
    	else
    		throw new IllegalArgumentException("Numpy dtype '" + type.getClass() + "' is not "
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
	 * @param fortranOrder
	 * 		  whether the numpy array was saved in fortran order or not (C-order)
     * @return The Img built from the tensor.
     * @throws IllegalArgumentException
     *         If the tensor type is not supported.
     */
    @SuppressWarnings("unchecked")
    public static <T extends NativeType<T>> RandomAccessibleInterval<T> build(ByteBuffer buf, ByteOrder byteOrder, String dtype, long[] shape, boolean fortranOrder) throws IllegalArgumentException
    {
    	long[] transposedShape = new long[shape.length];
    	for (int i = 0; i < shape.length; i ++)
    		transposedShape[i] = shape[shape.length - i - 1];
    	if (dtype.equals("int8") && !fortranOrder) {
    		ByteAccess access = new ByteBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) Utils.transpose(ArrayImgs.bytes( access, transposedShape ));
    	} else if (dtype.equals("int8")) {
    		ByteAccess access = new ByteBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) ArrayImgs.bytes( access, shape );
    	} else if (dtype.equals("uint8") && !fortranOrder) {
    		ByteAccess access = new ByteBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) Utils.transpose(ArrayImgs.unsignedBytes( access, transposedShape ));
    	} else if (dtype.equals("uint8")) {
    		ByteAccess access = new ByteBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) ArrayImgs.unsignedBytes( access, shape );
    	} else if (dtype.equals("int16") && !fortranOrder) {
    		ShortAccess access = new ShortBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) Utils.transpose(ArrayImgs.shorts( access, transposedShape ));
    	} else if (dtype.equals("int16")) {
    		ShortAccess access = new ShortBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) ArrayImgs.shorts( access, shape );
    	} else if (dtype.equals("uint16") && !fortranOrder) {
    		ShortAccess access = new ShortBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) Utils.transpose(ArrayImgs.unsignedShorts( access, transposedShape ));
    	} else if (dtype.equals("uint16")) {
    		ShortAccess access = new ShortBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) ArrayImgs.unsignedShorts( access, shape );
    	} else if (dtype.equals("int32") && !fortranOrder) {
    		IntAccess access = new IntBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) Utils.transpose(ArrayImgs.ints( access, transposedShape ));
    	} else if (dtype.equals("int32")) {
    		IntAccess access = new IntBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) ArrayImgs.ints( access, shape );
    	} else if (dtype.equals("uint32") && !fortranOrder) {
    		IntAccess access = new IntBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) Utils.transpose(ArrayImgs.unsignedInts( access, transposedShape ));
    	} else if (dtype.equals("uint32")) {
    		IntAccess access = new IntBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) ArrayImgs.unsignedInts( access, shape );
    	} else if (dtype.equals("int64") && !fortranOrder) {
    		LongAccess access = new LongBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) Utils.transpose(ArrayImgs.longs( access, transposedShape ));
    	} else if (dtype.equals("int64")) {
    		LongAccess access = new LongBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) ArrayImgs.longs( access, shape );
    	} else if (dtype.equals("float32") && !fortranOrder) {
    		FloatAccess access = new FloatBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) Utils.transpose(ArrayImgs.floats( access, transposedShape ));
    	} else if (dtype.equals("float32")) {
    		FloatAccess access = new FloatBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) ArrayImgs.floats( access, shape );
    	} else if (dtype.equals("float64") && !fortranOrder) {
    		DoubleAccess access = new DoubleBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) Utils.transpose(ArrayImgs.doubles( access, transposedShape ));
    	} else if (dtype.equals("float64")) {
    		DoubleAccess access = new DoubleBufferAccess(buf, true);
    		return (RandomAccessibleInterval<T>) ArrayImgs.doubles( access, shape );
    	} else if (dtype.equals("bool") && !fortranOrder) {
    		return (RandomAccessibleInterval<T>) Utils.transpose(buildBoolean(buf, byteOrder, transposedShape));
    	} else if (dtype.equals("bool")) {
    		return (RandomAccessibleInterval<T>) buildBoolean(buf, byteOrder, shape);
    	} else {
            throw new IllegalArgumentException("Unsupported data type of numpy array: " + dtype);
    	}
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
    
    /**
     * Method to convert a {@link RandomAccessibleInterval} into the byte array that is used by Numpy
     * to create .npy files.
     * The byte array created contains the flattened data of the {@link RandomAccessibleInterval} plus
     * information of the shape, data type, fortran order and byte order
     * @param <T>
     * 	possible ImgLib2 data types of the provided {@link RandomAccessibleInterval}
     * @param rai
     * 	the {@link RandomAccessibleInterval} of interest (an n-dimensional array) that is going to be
     * 	converted into the byte array
     * @return a byte array containing all the info to recreate it. An array in the format of Numpy .npy
     */
    public static < T extends RealType< T > & NativeType< T > >  byte[]
    createNumpyStyleByteArray(RandomAccessibleInterval<T> rai) {
    	String strHeader = "{'descr': '<";
    	strHeader += getDataType(rai.getAt(rai.minAsLongArray()));
    	strHeader += "', 'fortran_order': False, 'shape': (";
    	for (long ll : rai.dimensionsAsLongArray()) strHeader += ll + ", ";
    	strHeader = strHeader.substring(0, strHeader.length() - 2);
    	strHeader += "), }" + System.lineSeparator();
    	byte[] bufInverse = strHeader.getBytes(StandardCharsets.UTF_8);
    	byte[] major = {1};
        byte[] minor = {0};
        byte[] len = new byte[2];
        len[0] = (byte) (short) strHeader.length();
        len[1] = (byte) (((short) strHeader.length()) >> 8);
        byte[] array;
        if (Util.getTypeFromInterval(rai) instanceof ByteType) {
        	byte[] data = (byte[]) ImgLib2ToArray.build(rai);
        	ByteBuffer byteBuffer = ByteBuffer.allocate(data.length).order(ByteOrder.LITTLE_ENDIAN);
            byteBuffer.put(data);
            array = byteBuffer.array();
        } else if (Util.getTypeFromInterval(rai) instanceof UnsignedByteType) {
        	byte[] data = (byte[]) ImgLib2ToArray.build(rai);
        	ByteBuffer byteBuffer = ByteBuffer.allocate(data.length).order(ByteOrder.LITTLE_ENDIAN);
            byteBuffer.put(data);
            array = byteBuffer.array();
        } else if (Util.getTypeFromInterval(rai) instanceof ShortType) {
        	short[] data = (short[]) ImgLib2ToArray.build(rai);
        	ByteBuffer byteBuffer = ByteBuffer.allocate(data.length * 2).order(ByteOrder.LITTLE_ENDIAN);        
            ShortBuffer intBuffer = byteBuffer.asShortBuffer();
            intBuffer.put(data);
            array = byteBuffer.array();
        } else if (Util.getTypeFromInterval(rai) instanceof UnsignedShortType) {
        	short[] data = (short[]) ImgLib2ToArray.build(rai);
        	ByteBuffer byteBuffer = ByteBuffer.allocate(data.length * 2).order(ByteOrder.LITTLE_ENDIAN);        
            ShortBuffer intBuffer = byteBuffer.asShortBuffer();
            intBuffer.put(data);
            array = byteBuffer.array();
        } else if (Util.getTypeFromInterval(rai) instanceof IntType) {
        	int[] data = (int[]) ImgLib2ToArray.build(rai);
        	ByteBuffer byteBuffer = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);       
        	IntBuffer intBuffer = byteBuffer.asIntBuffer();
            intBuffer.put(data);
            array = byteBuffer.array();
        } else if (Util.getTypeFromInterval(rai) instanceof UnsignedIntType) {
        	int[] data = (int[]) ImgLib2ToArray.build(rai);
        	ByteBuffer byteBuffer = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);       
            IntBuffer intBuffer = byteBuffer.asIntBuffer();
            intBuffer.put(data);
            array = byteBuffer.array();
        } else if (Util.getTypeFromInterval(rai) instanceof LongType) {
        	long[] data = (long[]) ImgLib2ToArray.build(rai);
        	ByteBuffer byteBuffer = ByteBuffer.allocate(data.length * 8).order(ByteOrder.LITTLE_ENDIAN);       
            LongBuffer intBuffer = byteBuffer.asLongBuffer();
            intBuffer.put(data);
            array = byteBuffer.array();
        } else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
        	float[] data = (float[]) ImgLib2ToArray.build(rai);
        	ByteBuffer byteBuffer = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);       
            FloatBuffer intBuffer = byteBuffer.asFloatBuffer();
            intBuffer.put(data);
            array = byteBuffer.array();
        } else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
        	double[] data = (double[]) ImgLib2ToArray.build(rai);
        	ByteBuffer byteBuffer = ByteBuffer.allocate(data.length * 8).order(ByteOrder.LITTLE_ENDIAN);       
            DoubleBuffer intBuffer = byteBuffer.asDoubleBuffer();
            intBuffer.put(data);
            array = byteBuffer.array();
        } else {
        	throw new IllegalArgumentException("Unsupported data type");
        }
        int totalLen = MAGIC_PREFIX.length + 2 + 2 + bufInverse.length + array.length;
        byte[] total = new byte[totalLen];
        int c = 0;
        for (int i = 0; i < MAGIC_PREFIX.length; i ++)
        	total[c ++] = MAGIC_PREFIX[i];
        total[c ++] = major[0];
        total[c ++] = minor[0];
        total[c ++] = len[0];
        total[c ++] = len[1];
        for (int i = 0; i < bufInverse.length; i ++)
        	total[c ++] = bufInverse[i];
        for (int i = 0; i < array.length; i ++)
        	total[c ++] = array[i];
        return total;
    }
    
    /**
     * Method that saves a {@link RandomAccessibleInterval} nd array to a .npy Numpy radable file
     * @param <T>
     * 	possible ImgLib2 data types of the provided {@link RandomAccessibleInterval}
     * @param filePath
     * 	path where the file will be saved
     * @param rai
     * 	the {@link RandomAccessibleInterval} of interest (an n-dimensional array) that is going to be
     * 	converted into the byte array
     * @throws FileNotFoundException if the file path provided is invalid
     * @throws IOException if there is any error saving the file
     */
    public static < T extends RealType< T > & NativeType< T > > 
    void writeRaiToNpyFile(String filePath, RandomAccessibleInterval<T> rai) throws FileNotFoundException, IOException {
    	byte[] total = createNumpyStyleByteArray(rai);
        try (FileOutputStream fos = new FileOutputStream(filePath)) {
            fos.write(total);
        }
    }
}
