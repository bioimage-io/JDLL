package org.bioimageanalysis.icy.deeplearning.numpy;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.LongStream;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

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
        long[] shape;
        long nPositions = 1;
        if (shapeStr.isEmpty()) {
            shape = new long[0];
            nPositions = 0;
        } else {
            String[] tokens = shapeStr.split(", ?");
            shape = Arrays.stream(tokens).mapToLong(Long::parseLong).toArray();
            LongStream.range(0, shape.length).map(i -> nPositions *= shape[(int) i] );
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

        return manager.create(dataType.asDataType(data), shape, dataType);
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

}
