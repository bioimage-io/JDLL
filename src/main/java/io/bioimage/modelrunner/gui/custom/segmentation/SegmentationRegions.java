package io.bioimage.modelrunner.gui.custom.segmentation;

import java.awt.Rectangle;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.BooleanSupplier;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

/** Pixel-exact regions in the source image's zero-based X,Y,Z,T coordinates. */
public final class SegmentationRegions {
    public enum Kind {
        INSTANCE_PLANES(false, false), INSTANCE_VOLUME(false, true),
        SEMANTIC_PLANES(true, false), SEMANTIC_VOLUME(true, true);
        public final boolean semantic, volume;
        Kind(boolean semantic, boolean volume) { this.semantic = semantic; this.volume = volume; }
    }

    /** Row-major triples (nonzero uint32 label bits, linear start, length), sorted, maximal and row bounded. */
    @FunctionalInterface
    public interface PlaneRuns { int[] read(int z, int t) throws IOException; }

    public static final class Slice {
        public final int z;
        private final int width;
        private final Ints spans = new Ints();
        private int left = Integer.MAX_VALUE, top = Integer.MAX_VALUE, right, bottom;
        private Slice(int z, int width) { this.z = z; this.width = width; }
        private void add(int start, int length) {
            spans.add(start); spans.add(length);
            int x = start % width, y = start / width;
            left = Math.min(left, x); top = Math.min(top, y);
            right = Math.max(right, x + length); bottom = y + 1;
        }
        public Rectangle bounds() { return new Rectangle(left, top, right - left, bottom - top); }
        public int runCount() { return spans.size / 2; }
        public int x(int i) { return spans.values[i * 2] % width; }
        public int y(int i) { return spans.values[i * 2] / width; }
        public int length(int i) { return spans.values[i * 2 + 1]; }
        public boolean[] croppedMask() {
            Rectangle b = bounds();
            boolean[] mask = new boolean[Math.multiplyExact(b.width, b.height)];
            for (int i = 0; i < runCount(); i++) {
                int start = (y(i) - b.y) * b.width + x(i) - b.x;
                Arrays.fill(mask, start, start + length(i), true);
            }
            return mask;
        }
    }

    public static final class Region {
        public final long label;
        public final int time, component;
        public final String id, name;
        private final List<Slice> slices = new ArrayList<>();
        private Region(long label, int z, int t, Kind kind, int component) {
            this.label = label; this.time = t; this.component = component;
            name = kind.semantic ? "Class " + label + " · region " + component : "Label " + label;
            id = "t" + t + (kind.volume ? "" : "-z" + z) + "-label" + label
                    + (kind.semantic ? "-region" + component : "");
        }
        public List<Slice> slices() { return Collections.unmodifiableList(slices); }
        private void add(int z, int width, int start, int length) {
            if (slices.isEmpty() || slices.get(slices.size() - 1).z != z) slices.add(new Slice(z, width));
            slices.get(slices.size() - 1).add(start, length);
        }
    }

    public final int width, height, depth, frames;
    public final Kind kind;
    private final List<Region> regions = new ArrayList<>();
    public List<Region> objects() { return Collections.unmodifiableList(regions); }
    private SegmentationRegions(int width, int height, int depth, int frames, Kind kind) {
        if (width < 1 || height < 1 || depth < 1 || frames < 1) throw new IllegalArgumentException("Empty label geometry");
        Math.multiplyExact(width, height);
        this.width = width; this.height = height; this.depth = depth; this.frames = frames; this.kind = kind;
    }

    public static <T extends RealType<T> & NativeType<T>> SegmentationRegions fromLabels(
            Tensor<T> tensor, Kind kind, BooleanSupplier cancelled) throws IOException {
        RandomAccessibleInterval<T> image = Views.zeroMin(tensor.getData());
        String axes = tensor.getAxesOrderString().toLowerCase(Locale.ROOT);
        int x = axes.indexOf('x'), y = axes.indexOf('y'), z = axes.indexOf('z');
        int t = axes.indexOf('t') >= 0 ? axes.indexOf('t') : axes.indexOf('b');
        if (x < 0 || y < 0 || axes.length() != image.numDimensions()) throw new IOException("Invalid label axes: " + axes);
        for (int i = 0; i < axes.length(); i++)
            if (i != x && i != y && i != z && i != t && image.dimension(i) != 1)
                throw new IOException("Expected a single label channel, got axes " + axes);
        int w = dim(image, x), h = dim(image, y);
        return fromRuns(w, h, dim(image, z), dim(image, t), kind, (slice, frame) -> {
            RandomAccessibleInterval<T> plane = image;
            for (int i = axes.length() - 1; i >= 0; i--)
                if (i != x && i != y) plane = Views.hyperSlice(plane, i, i == z ? slice : i == t ? frame : 0);
            if (x > y) plane = Views.permute(plane, 0, 1);
            Cursor<T> pixels = Views.flatIterable(plane).cursor();
            Ints runs = new Ints();
            for (int row = 0; row < h; row++) {
                if ((row & 63) == 0) check(cancelled);
                int current = 0, start = 0;
                for (int col = 0; col < w; col++) {
                    double value = pixels.next().getRealDouble();
                    long label = (long) value;
                    if (label != value || label < 0 || label > 0xffffffffL)
                        throw new IOException("Expected nonnegative 32-bit integer labels, found " + value);
                    int id = (int)label;
                    if (id == current) continue;
                    if (current != 0) runs.triple(current, row * w + start, col - start);
                    current = id; start = col;
                }
                if (current != 0) runs.triple(current, row * w + start, w - start);
            }
            return Arrays.copyOf(runs.values, runs.size);
        }, cancelled);
    }

    private static int dim(RandomAccessibleInterval<?> image, int axis) {
        return axis < 0 ? 1 : Math.toIntExact(image.dimension(axis));
    }

    public static SegmentationRegions fromRuns(int width, int height, int depth, int frames, Kind kind,
            PlaneRuns source, BooleanSupplier cancelled) throws IOException {
        SegmentationRegions result = new SegmentationRegions(width, height, depth, frames, kind);
        int scope = kind.volume ? depth : 1;
        for (int t = 0; t < frames; t++) {
            for (int firstZ = 0; firstZ < depth; firstZ += scope) {
                List<Plane> planes = new ArrayList<>();
                Union groups = new Union();
                Map<Integer, Region> objects = new LinkedHashMap<>();
                for (int z = firstZ; z < firstZ + scope; z++) {
                    check(cancelled);
                    Plane plane = new Plane(source.read(z, t), width, height);
                    if (kind.semantic) {
                        plane.first = groups.allocate(plane.count());
                        for (int y = 0; y < height; y++) {
                            if ((y & 255) == 0) check(cancelled);
                            if (y > 0) join(plane, y, plane, y - 1, width, groups);
                            if (!planes.isEmpty()) join(plane, y, planes.get(planes.size() - 1), y, width, groups);
                        }
                        planes.add(plane);
                    } else {
                        result.collect(plane, objects, null, z, t, cancelled);
                    }
                }
                for (int i = 0; i < planes.size(); i++)
                    result.collect(planes.get(i), objects, groups, firstZ + i, t, cancelled);
                result.regions.addAll(objects.values());
            }
        }
        return result;
    }

    private void collect(Plane p, Map<Integer, Region> objects, Union groups, int z, int t,
            BooleanSupplier cancelled) throws IOException {
        for (int i = 0; i < p.count(); i++) {
            if ((i & 4095) == 0) check(cancelled);
            int label = p.data[3*i], key = groups == null ? label : groups.root(p.first + i);
            Region object = objects.get(key);
            if (object == null) {
                object = new Region(Integer.toUnsignedLong(label), z, t, kind, groups == null ? 0 : objects.size() + 1);
                objects.put(key, object);
            }
            object.add(z, width, p.data[3*i+1], p.data[3*i+2]);
        }
    }

    private static final class Plane {
        final int[] data, rows;
        int first;
        int count() { return data.length / 3; }
        Plane(int[] data, int width, int height) throws IOException {
            if (data == null || data.length % 3 != 0) throw new IOException("Incomplete label runs");
            this.data = data;
            rows = new int[height + 1];
            int nextRow = 0, previousEnd = 0;
            for (int i = 0; i < count(); i++) {
                int id = data[3*i], p = data[3*i+1], n = data[3*i+2];
                if (id == 0 || p < previousEnd || p / width >= height || n <= 0 || n > width - p % width)
                    throw new IOException("Invalid label run at index " + i);
                if (i > 0 && p == previousEnd && p % width != 0 && id == data[3*i-3])
                    throw new IOException("Adjacent runs of one label must be joined");
                previousEnd = p + n;
                while (nextRow <= p / width) rows[nextRow++] = i;
            }
            while (nextRow < rows.length) rows[nextRow++] = count();
        }
    }

    /** Overlapping same-class spans across face-adjacent rows or planes implement 4/6 connectivity. */
    private static void join(Plane a, int ay, Plane b, int by, int width, Union groups) {
        int i = a.rows[ay], j = b.rows[by];
        while (i < a.rows[ay+1] && j < b.rows[by+1]) {
            int ax = a.data[3*i+1] % width, bx = b.data[3*j+1] % width;
            int ae = ax + a.data[3*i+2], be = bx + b.data[3*j+2];
            if (ax < be && bx < ae && a.data[3*i] == b.data[3*j]) groups.join(a.first + i, b.first + j);
            if (ae <= be) i++;
            if (be <= ae) j++;
        }
    }

    static final class Ints {
        int[] values = new int[12];
        int size;
        void add(int n) {
            if (size == values.length) values = Arrays.copyOf(values, Math.multiplyExact(size, 2));
            values[size++] = n;
        }
        void triple(int a, int b, int c) { add(a); add(b); add(c); }
    }
    private static final class Union {
        final Ints parents = new Ints();
        int allocate(int count) {
            int start = parents.size;
            for (int i = 0; i < count; i++) parents.add(-1);
            return start;
        }
        int root(int i) {
            int r = i;
            while (parents.values[r] >= 0) r = parents.values[r];
            while (r != i) { int p = parents.values[i]; parents.values[i] = r; i = p; }
            return r;
        }
        void join(int a, int b) {
            a = root(a); b = root(b);
            if (a == b) return;
            if (parents.values[a] > parents.values[b]) { int swap = a; a = b; b = swap; }
            parents.values[a] += parents.values[b]; parents.values[b] = a;
        }
    }

    public static void check(BooleanSupplier cancelled) throws InterruptedIOException {
        if (Thread.currentThread().isInterrupted() || cancelled.getAsBoolean()) throw new InterruptedIOException("Segmentation output cancelled.");
    }
}
