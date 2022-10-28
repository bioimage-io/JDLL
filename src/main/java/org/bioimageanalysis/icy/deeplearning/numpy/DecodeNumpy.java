package org.bioimageanalysis.icy.deeplearning.numpy;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.LongStream;

import org.bioimageanalysis.icy.deeplearning.utils.IndexingUtils;

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
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

public class DecodeNumpy {
	
    private static final int VERSION = 3;

    private static final int BUFFER_SIZE = 1024 * 1024;
    private static final String MAGIC_NUMBER = "NDAR";
    private static final byte[] NUMPY_HEADER = {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'};
    private static final int ARRAY_ALIGN = 64;

    private static final Pattern HEADER_PATTERN =
            Pattern.compile("\\{'descr': '(.+)', 'fortran_order': False, 'shape': \\((.*)\\),");
	


    static < T extends RealType< T > & NativeType< T > > 
    				RandomAccessibleInterval<T> decodeNumpy(InputStream is) throws IOException {
        DataInputStream dis;
        if (is instanceof DataInputStream) {
            dis = (DataInputStream) is;
        } else {
            dis = new DataInputStream(is);
        }

        byte[] buf = new byte[NUMPY_HEADER.length];
        dis.readFully(buf);
        if (!Arrays.equals(buf, NUMPY_HEADER)) {
            throw new IllegalArgumentException("Malformed numpy data");
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
        long numBytes = 1;
        if (typeStr.toLowerCase().equals("float32")) {
        	numBytes = 4;
        } else if (typeStr.toLowerCase().equals("int32")) {
        	numBytes = 4;
        } else if (typeStr.toLowerCase().equals("float64")) {
        	numBytes = 8;
        } else if (typeStr.toLowerCase().equals("byte")) {
        	numBytes = 1;
        } else if (typeStr.toLowerCase().equals("int64")) {
        	numBytes = 8;
        }
        len = Math.toIntExact(shape.length * numBytes);
        ByteBuffer data = ByteBuffer.allocate(len);
        char order = typeStr.charAt(0);
        if (order == '>') {
            data.order(ByteOrder.BIG_ENDIAN);
        } else if (order == '<') {
            data.order(ByteOrder.LITTLE_ENDIAN);
        }
        readData(dis, data, len);

        return build(data, typeStr, shape);
    }

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
     * Creates a {@link Img} from a given {@link Tensor} and an array with its dimensions order.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor.
     * @throws IllegalArgumentException
     *         If the tensor type is not supported.
     */
    @SuppressWarnings("unchecked")
    public static <T extends Type<T>> Img<T> build(ByteBuffer buf, String typeStr, long[] shape) throws IllegalArgumentException
    {
        // Create an INDArray of the same type of the tensor
    	byte[] data = new byte[buf.remaining()];
    	buf.get(data);
    	if (typeStr.equals("byte")) {
            return (Img<T>) buildFromTensorByte(data, shape);
    	} else if (typeStr.equals("int32")) {
            return (Img<T>) buildFromTensorInt(data, shape);
    	} else if (typeStr.equals("ubyte")) {
            return (Img<T>) buildFromTensoUByte(data, shape);
    	} else if (typeStr.equals("float32")) {
            return (Img<T>) buildFromTensorFloat(data, shape);
    	} else if (typeStr.equals("float64")) {
            return (Img<T>) buildFromTensorDouble(data, shape);
    	} else if (typeStr.equals("int64")) {
            return (Img<T>) buildFromTensorLong(data, shape);
    	} else if (typeStr.equals("float32")) {
            return (Img<T>) buildFromTensorFloat(data, shape);
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + typeStr);
    	}
    }

    /** TODO check BigEndian LittleEndian
     * Builds a {@link Img} from a unsigned byte-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#UBYTE}.
     */
    private static Img<ByteType> buildFromTensorByte(byte[] tensor, ByteOrder byteOrder, long[] tensorShape)
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
     * Builds a {@link Img} from a unsigned integer-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The sequence built from the tensor of type {@link DataType#INT}.
     */
    private static Img<IntType> buildFromTensorInt(byte[] tensor, long[] tensorShape)
    {
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
        final Img< IntType > outputImg = (Img<IntType>) factory.create(tensorShape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
        int[] flatArr = ByteArrayUtils.convertIntoSignedInt32(tensor);
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
     * Builds a {@link Img} from a unsigned float-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#FLOAT}.
     */
    private static Img<FloatType> buildFromTensorFloat(byte[] tensor , long[] tensorShape)
    {
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(tensorShape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
        float[] flatArr = ByteArrayUtils.convertIntoSignedFloat32(tensor);
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
     * Builds a {@link Img} from a unsigned double-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<DoubleType> buildFromTensorDouble(byte[] tensor, long[] tensorShape)
    {
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(tensorShape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
		double[] flatArr = ByteArrayUtils.convertIntoSignedFloat64(tensor);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	double val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

}
