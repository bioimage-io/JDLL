package io.bioimage.modelrunner.gui.custom.segmentation;

import java.awt.Polygon;
import java.awt.Rectangle;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BooleanSupplier;

import com.google.gson.stream.JsonWriter;
import io.bioimage.modelrunner.gui.custom.segmentation.SegmentationRegions.Region;
import io.bioimage.modelrunner.gui.custom.segmentation.SegmentationRegions.Slice;

/** GeoJSON in image pixel-edge coordinates. Z and T are zero-based feature properties. */
public final class SegmentationGeoJson {
    private SegmentationGeoJson() {}

    public static void write(File target, File source, SegmentationRegions regions, BooleanSupplier cancelled)
            throws IOException {
        try (JsonWriter out = new JsonWriter(Files.newBufferedWriter(target.toPath(), StandardCharsets.UTF_8))) {
            out.beginObject().name("type").value("FeatureCollection");
            out.name("properties").beginObject().name("image_path").value(source.getAbsolutePath())
               .name("coordinate_system").value("pixel").name("origin").value("top-left")
               .name("width").value(regions.width).name("height").value(regions.height)
               .name("depth").value(regions.depth).name("frames").value(regions.frames).endObject();
            out.name("features").beginArray();
            for (Region region : regions.objects()) {
                for (Slice slice : region.slices()) {
                    SegmentationRegions.check(cancelled);
                    List<Ring> polygons = new Tracer(slice).polygons(cancelled);
                    out.beginObject().name("type").value("Feature").name("id").value(region.id + "/z" + slice.z);
                    out.name("properties").beginObject().name("object_id").value(region.id)
                       .name("name").value(region.name).name("label").value(region.label)
                       .name("z").value(slice.z).name("t").value(region.time);
                    if (regions.kind.semantic) out.name("class_id").value(region.label).name("component").value(region.component);
                    out.endObject().name("geometry").beginObject().name("type")
                       .value(polygons.size() == 1 ? "Polygon" : "MultiPolygon").name("coordinates");
                    if (polygons.size() > 1) out.beginArray();
                    for (Ring shell : polygons) {
                        out.beginArray();
                        writeRing(out, shell);
                        for (Ring hole : shell.holes) writeRing(out, hole);
                        out.endArray();
                    }
                    if (polygons.size() > 1) out.endArray();
                    out.endObject().endObject();
                }
            }
            out.endArray().endObject();
        }
    }

    private static void writeRing(JsonWriter out, Ring ring) throws IOException {
        out.beginArray();
        Polygon p = ring.polygon;
        for (int i = 0; i <= p.npoints; i++) {
            int j = i % p.npoints;
            out.beginArray().value(p.xpoints[j]).value(p.ypoints[j]).endArray();
        }
        out.endArray();
    }

    private static final class Ring {
        final Polygon polygon;
        final double area;
        final List<Ring> holes = new ArrayList<>();
        Ring(Polygon p) {
            polygon = p;
            double sum = 0;
            for (int i = 0, j = p.npoints - 1; i < p.npoints; j = i++)
                sum += (double)(p.xpoints[j] - p.xpoints[0]) * (p.ypoints[i] - p.ypoints[0])
                     - (double)(p.xpoints[i] - p.xpoints[0]) * (p.ypoints[j] - p.ypoints[0]);
            area = sum / 2;
        }
    }

    private static final class Tracer {
        static final int[] DX = {1,0,-1,0}, DY = {0,1,0,-1};
        static final int[] RX = {0,-1,-1,0}, RY = {0,0,-1,-1};
        static final int[] LX = {0,0,-1,-1}, LY = {-1,0,0,-1};
        final Slice slice;
        final Rectangle bounds;
        final boolean[] mask, seen;
        Tracer(Slice slice) {
            this.slice = slice;
            bounds = slice.bounds();
            long pixels = 0;
            for (int i = 0; i < slice.runCount(); i++) pixels += slice.length(i);
            // Widely separated parts of one label can share an enormous bounding rectangle.
            mask = (long) bounds.width * bounds.height <= Math.max(256, pixels * 8) ? slice.croppedMask() : null;
            seen = new boolean[slice.runCount()];
        }

        int runAt(int x, int y) {
            int lo = 0, hi = slice.runCount() - 1;
            while (lo <= hi) {
                int mid = (lo + hi) >>> 1, row = slice.y(mid), start = slice.x(mid);
                if (row > y || (row == y && start > x)) hi = mid - 1;
                else if (row < y || x >= start + slice.length(mid)) lo = mid + 1;
                else return mid;
            }
            return -1;
        }

        boolean contains(int x, int y) {
            if (!bounds.contains(x, y)) return false;
            return mask == null ? runAt(x, y) >= 0 : mask[(y - bounds.y) * bounds.width + x - bounds.x];
        }

        List<Ring> polygons(BooleanSupplier cancelled) throws IOException {
            List<Ring> rings = new ArrayList<>();
            for (int i = 0; i < seen.length; i++) {
                if ((i & 1023) == 0) SegmentationRegions.check(cancelled);
                if (seen[i]) continue;
                int x0 = slice.x(i) + slice.length(i), y0 = slice.y(i);
                int x = x0, y = y0, dir = 1, steps = 0;
                boolean saddle = false;
                Polygon path = new Polygon();
                do {
                    if ((++steps & 16383) == 0) SegmentationRegions.check(cancelled);
                    if (dir == 1) seen[runAt(x - 1, y)] = true;
                    x += DX[dir]; y += DY[dir];
                    int next = dir;
                    if (!contains(x + RX[dir], y + RY[dir])) {
                        saddle |= contains(x + LX[dir], y + LY[dir]);
                        next = (dir + 1) & 3;
                    } else if (contains(x + LX[dir], y + LY[dir])) next = (dir + 3) & 3;
                    if (next != dir) path.addPoint(x, y);
                    dir = next;
                } while (x != x0 || y != y0 || dir != 1);
                if (saddle) splitAtRepeatedCorners(path, rings);
                else rings.add(new Ring(path));
            }
            List<Ring> shells = new ArrayList<>();
            for (Ring r : rings) if (r.area > 0) shells.add(r);
            for (Ring hole : rings) {
                if (hole.area >= 0) continue;
                // Sample a quarter pixel into foreground at the first edge. A nested island cannot contain it.
                int[] xs = hole.polygon.xpoints, ys = hole.polygon.ypoints;
                double x = xs[0] + .5 * (xs[1] - xs[0]) - .25 * Integer.signum(ys[1] - ys[0]);
                double y = ys[0] + .5 * (ys[1] - ys[0]) + .25 * Integer.signum(xs[1] - xs[0]);
                Ring parent = null;
                for (Ring shell : shells)
                    if ((parent == null || shell.area < parent.area) && shell.polygon.contains(x, y)) parent = shell;
                if (parent == null) throw new IOException("Could not locate the exterior of a label hole");
                parent.holes.add(hole);
            }
            return shells;
        }
    }

    // A four-connected outline can revisit a diagonal pinch. Split its walk into simple linear rings.
    private static void splitAtRepeatedCorners(Polygon path, List<Ring> out) {
        Map<Long,Integer> positions = new HashMap<>();
        int[] stack = new int[path.npoints];
        int size = 0;
        for (int i = 0; i <= path.npoints; i++) {
            int vertex = i % path.npoints;
            long key = key(path, vertex);
            Integer at = positions.get(key);
            if (at == null) {
                positions.put(key, size);
                stack[size++] = vertex;
            } else {
                Polygon ring = new Polygon();
                for (int j = at; j < size; j++) {
                    int v = stack[j];
                    ring.addPoint(path.xpoints[v], path.ypoints[v]);
                    if (j > at) positions.remove(key(path, v));
                }
                out.add(new Ring(ring));
                size = at + 1;
            }
        }
    }

    private static long key(Polygon path, int i) {
        return ((long)path.xpoints[i] << 32) | (path.ypoints[i] & 0xffffffffL);
    }
}
