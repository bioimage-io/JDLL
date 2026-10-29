package io.bioimage.modelrunner.numpy;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.StandardOpenOption;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Header-only inspection and bounded read-only pixel access for numeric NumPy images. */
public final class NpyImageFile implements AutoCloseable {
    public static final String DOCUMENTATION_URL = "https://github.com/bioimage-io/JDLL/blob/main/docs/image-formats.md";
    private static final int WINDOW = 16 * 1024 * 1024;
    private final FileChannel channel;
    private final File file;
    private final long offset;
    private final long[] shape;
    private final long[] strides;
    private final char kind;
    private final int bytes;
    private final ByteOrder order;
    private ByteBuffer buffer;
    private long bufferStart = -1;
    private String axes;

    public static boolean isNpy(File file) {
        return file.getName().toLowerCase(java.util.Locale.ROOT).endsWith(".npy");
    }

    public NpyImageFile(File file) throws IOException {
        this.file = file;
        String header;
        long start;
        try (DataInputStream in = new DataInputStream(java.nio.file.Files.newInputStream(file.toPath()))) {
            byte[] magic = new byte[6];
            in.readFully(magic);
            if (!java.util.Arrays.equals(magic, new byte[] {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'}))
                throw invalid("Not a NumPy array");
            int major = in.readUnsignedByte(), minor = in.readUnsignedByte();
            if (major < 1 || major > 3 || minor != 0) throw invalid("Unsupported NumPy version");
            int n = major == 1 ? 2 : 4;
            long length = 0;
            for (int i = 0; i < n; i++) length |= (long) in.readUnsignedByte() << (8 * i);
            if (length < 1 || length > 65536) throw invalid("Unsupported NumPy header size");
            byte[] text = new byte[(int) length];
            in.readFully(text);
            header = new String(text, StandardCharsets.UTF_8);
            start = 8 + n + length;
        } catch (java.io.EOFException e) {
            throw invalid("Truncated NumPy header");
        }
        Matcher descriptor = Pattern.compile("['\"]descr['\"]\\s*:\\s*['\"]([<>=|])([biuf])(\\d+)['\"]").matcher(header);
        Matcher dimensions = Pattern.compile("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)").matcher(header);
        Matcher fortran = Pattern.compile("['\"]fortran_order['\"]\\s*:\\s*(True|False)").matcher(header);
        if (!descriptor.find() || !dimensions.find() || !fortran.find()) throw invalid("Expected a plain numeric array, not objects or structured records");
        kind = descriptor.group(2).charAt(0);
        bytes = Integer.parseInt(descriptor.group(3));
        if (!(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8) || kind == 'f' && bytes != 4 && bytes != 8
                || kind == 'b' && bytes != 1)
            throw invalid("Unsupported numeric datatype: " + descriptor.group());
        order = ">".equals(descriptor.group(1)) ? ByteOrder.BIG_ENDIAN
                : "=".equals(descriptor.group(1)) ? ByteOrder.nativeOrder() : ByteOrder.LITTLE_ENDIAN;
        String[] parts = dimensions.group(1).trim().split("\\s*,\\s*");
        shape = new long[parts.length];
        if (parts.length < 2 || parts.length > 5) throw invalid("Unsupported image type: expected 2 to 5 dimensions");
        try {
            for (int i = 0; i < parts.length; i++) {
                shape[i] = Long.parseLong(parts[i]);
                if (shape[i] <= 0) throw invalid("Empty dimensions are unsupported");
            }
            strides = new long[shape.length];
            long size = bytes;
            boolean f = "True".equals(fortran.group(1));
            for (int i = 0; i < shape.length; i++) {
                int axis = f ? i : shape.length - i - 1;
                strides[axis] = size;
                size = Math.multiplyExact(size, shape[axis]);
            }
            if (java.nio.file.Files.size(file.toPath()) < Math.addExact(start, size)) throw invalid("Truncated array data");
        } catch (NumberFormatException | ArithmeticException e) {
            throw invalid("Unsupported array shape");
        }
        offset = start;
        channel = FileChannel.open(file.toPath(), StandardOpenOption.READ);
    }

    /** Resolve XY[CZT] or [TZC]YX, optionally guided by the paired image geometry. */
    public NpyImageFile resolve(int width, int height) throws IOException {
        resolveVolume(width, height, -1, false);
        if (depth() != 1 || timePoints() != 1)
            throw invalid("YOLO training requires 2D samples; split non-singleton Z/T axes into annotated images");
        return this;
    }

    /** Resolve numeric image/label geometry without imposing YOLO's 2D restriction. */
    public NpyImageFile resolveVolume(int width, int height, int depth, boolean mask) throws IOException {
        String first = new String[] {"XY", "XYC", "XYCZ", "XYCZT"}[shape.length - 2];
        String last = new String[] {"YX", "CYX", "ZCYX", "TZCYX"}[shape.length - 2];
        java.util.List<String> candidates = new java.util.ArrayList<>();
        candidates.add(first);
        candidates.add(last);
        if (mask && shape.length == 3) java.util.Collections.addAll(candidates, "XYZ", "ZYX");
        if (mask && shape.length == 4) java.util.Collections.addAll(candidates, "XYZT", "TZYX");
        axes = null;
        for (String candidate : candidates) {
            if (!matches(candidate, width, height)) continue;
            int z = candidate.indexOf('Z');
            if (depth >= 0 && (z < 0 ? 1 : shape[z]) != depth) continue;
            if (axes == null || area(candidate) >= area(axes)) axes = candidate;
        }
        if (axes == null && mask && (width > 0 || height > 0)) return resolveVolume(0, 0, -1, true);
        if (axes == null) throw invalid("Cannot match XY/C/Z/T axes to the paired geometry for " + java.util.Arrays.toString(shape));
        if (timePoints() != 1) throw invalid("Training requires individual time points; split the non-singleton T axis");
        return this;
    }

    private boolean matches(String candidate, int width, int height) {
        long x = shape[candidate.indexOf('X')], y = shape[candidate.indexOf('Y')];
        if (width > 0 && height > 0) return x == width && y == height;
        return x > 100 && y > 100 && Math.max(x, y) / (double) Math.min(x, y) < 3;
    }

    private double area(String candidate) {
        return (double) shape[candidate.indexOf('X')] * shape[candidate.indexOf('Y')];
    }

    public int width() { return Math.toIntExact(shape[axes.indexOf('X')]); }
    public int height() { return Math.toIntExact(shape[axes.indexOf('Y')]); }
    public boolean isFloatingPoint() { return kind == 'f'; }
    public int channels() { return axes.indexOf('C') < 0 ? 1 : Math.toIntExact(shape[axes.indexOf('C')]); }
    public int depth() { return axes.indexOf('Z') < 0 ? 1 : Math.toIntExact(shape[axes.indexOf('Z')]); }
    public int timePoints() { return axes.indexOf('T') < 0 ? 1 : Math.toIntExact(shape[axes.indexOf('T')]); }
    public String axesOrder() { return axes; }

    public double value(int x, int y, int c) throws IOException {
        return value(x, y, 0, c);
    }

    public double value(int x, int y, int z, int c) throws IOException {
        long position = offset + x * strides[axes.indexOf('X')] + y * strides[axes.indexOf('Y')];
        if (axes.indexOf('Z') >= 0) position += z * strides[axes.indexOf('Z')];
        if (axes.indexOf('C') >= 0) position += c * strides[axes.indexOf('C')];
        if (buffer == null || position < bufferStart || position + bytes > bufferStart + buffer.limit()) {
            bufferStart = position - position % WINDOW;
            int length = (int) Math.min(WINDOW + 8L, channel.size() - bufferStart);
            if (buffer == null || buffer.capacity() < length) buffer = ByteBuffer.allocate(length).order(order);
            buffer.clear().limit(length);
            long p = bufferStart;
            while (buffer.hasRemaining()) {
                int read = channel.read(buffer, p);
                if (read < 0) throw invalid("Truncated array data");
                p += read;
            }
            buffer.flip();
        }
        int at = (int) (position - bufferStart);
        if (kind == 'f') return bytes == 4 ? buffer.getFloat(at) : buffer.getDouble(at);
        if (kind == 'b') return buffer.get(at) == 0 ? 0 : 1;
        if (bytes == 1) return kind == 'u' ? Byte.toUnsignedInt(buffer.get(at)) : buffer.get(at);
        if (bytes == 2) return kind == 'u' ? Short.toUnsignedInt(buffer.getShort(at)) : buffer.getShort(at);
        if (bytes == 4) return kind == 'u' ? Integer.toUnsignedLong(buffer.getInt(at)) : buffer.getInt(at);
        long value = buffer.getLong(at);
        return kind == 'u' && value < 0 ? (value & Long.MAX_VALUE) + 0x1.0p63 : value;
    }

    /** Scan a channel in storage order, avoiding repeated window reads for Fortran arrays. */
    public void forEachValue(int c, java.util.function.DoubleConsumer consumer) throws IOException {
        int[] sizes = {width(), height(), depth()}, position = new int[3];
        long[] steps = {strides[axes.indexOf('X')], strides[axes.indexOf('Y')],
                axes.indexOf('Z') < 0 ? Long.MAX_VALUE : strides[axes.indexOf('Z')]};
        Integer[] order = {0, 1, 2};
        java.util.Arrays.sort(order, java.util.Comparator.comparingLong(axis -> steps[axis]));
        for (position[order[2]] = 0; position[order[2]] < sizes[order[2]]; position[order[2]]++)
            for (position[order[1]] = 0; position[order[1]] < sizes[order[1]]; position[order[1]]++)
                for (position[order[0]] = 0; position[order[0]] < sizes[order[0]]; position[order[0]]++)
                    consumer.accept(value(position[0], position[1], position[2], c));
    }

    private IOException invalid(String reason) {
        return new IOException(reason + ": " + file.getAbsolutePath() + ". See " + DOCUMENTATION_URL);
    }

    @Override public void close() throws IOException { channel.close(); }
}
