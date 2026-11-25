package io.bioimage.modelrunner.gui.custom.segmentation;

import static org.junit.Assert.*;
import java.awt.geom.Path2D;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import com.google.gson.*;
import io.bioimage.modelrunner.gui.custom.InferenceRunFiles;
import io.bioimage.modelrunner.gui.custom.segmentation.SegmentationRegions.Kind;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.view.Views;

public class SegmentationOutputTest {
    @Rule public TemporaryFolder tmp = new TemporaryFolder();

    @Test public void allThreeByThreePatternsHaveExactSimplePixelEdgeRings() throws Exception {
        // Exercise all diagonal pinches, holes, disconnected parts and border contacts in one file.
        int w = 32 * 4, h = 16 * 4;
        int[] pixels = new int[w*h];
        for (int id = 1; id < 512; id++) for (int bit = 0; bit < 9; bit++)
            if ((id & (1 << bit)) != 0)
                pixels[(id / 32 * 4 + bit / 3)*w + id % 32 * 4 + bit % 3] = id;
        Tensor<IntType> labels = Tensor.build("labels", "yx", Views.permute(ArrayImgs.ints(pixels, w, h), 0, 1));
        SegmentationRegions regions = SegmentationRegions.fromLabels(labels, Kind.INSTANCE_PLANES, () -> false);
        assertEquals(511, regions.objects().size());
        File geo = tmp.newFile();
        SegmentationGeoJson.write(geo, geo, regions, () -> false);
        int[] observed = new int[w*h];
        try (Reader reader = Files.newBufferedReader(geo.toPath())) {
            JsonArray features = new JsonParser().parse(reader).getAsJsonObject().getAsJsonArray("features");
            assertEquals(511, features.size());
            for (JsonElement feature : features) {
                JsonObject f = feature.getAsJsonObject();
                int id = f.getAsJsonObject("properties").get("label").getAsInt();
                JsonObject geometry = f.getAsJsonObject("geometry");
                JsonArray coordinates = geometry.getAsJsonArray("coordinates");
                if (geometry.get("type").getAsString().equals("Polygon")) fill(coordinates, observed, w, id);
                else for (JsonElement polygon : coordinates) fill(polygon.getAsJsonArray(), observed, w, id);
            }
        }
        assertArrayEquals(pixels, observed);
    }

    private static void fill(JsonArray polygon, int[] image, int width, int id) {
        Path2D.Double path = new Path2D.Double(Path2D.WIND_EVEN_ODD);
        for (int r = 0; r < polygon.size(); r++) {
            JsonArray coords = polygon.get(r).getAsJsonArray();
            assertEquals(coords.get(0), coords.get(coords.size()-1));
            Set<String> visited = new HashSet<>();
            double area2 = 0;
            for (int i = 0; i < coords.size()-1; i++) {
                JsonArray p = coords.get(i).getAsJsonArray(), q = coords.get(i+1).getAsJsonArray();
                int x = p.get(0).getAsInt(), y = p.get(1).getAsInt();
                assertTrue("Repeated corner in label " + id, visited.add(x + "," + y));
                assertTrue((x == q.get(0).getAsInt()) != (y == q.get(1).getAsInt()));
                area2 += x*q.get(1).getAsInt() - y*q.get(0).getAsInt();
                if (i == 0) path.moveTo(x,y); else path.lineTo(x,y);
            }
            assertTrue(r == 0 ? area2 > 0 : area2 < 0);
            path.closePath();
        }
        java.awt.Rectangle bounds = path.getBounds();
        for (int y = bounds.y; y < bounds.y+bounds.height; y++)
            for (int x = bounds.x; x < bounds.x+bounds.width; x++) if (path.contains(x+.5,y+.5)) {
                assertEquals(0, image[y*width+x]);
                image[y*width+x] = id;
            }
    }

    @Test public void semanticAndInstanceIdentityRespectModelDimensionalityAndTime() throws Exception {
        int[] plane = {7,0,7, 0,9,0};
        int[] data = new int[24];
        for (int i=0; i<4; i++) System.arraycopy(plane,0,data,6*i,6);
        Tensor<IntType> labels = Tensor.build("labels","xyzt",ArrayImgs.ints(data,3,2,2,2));
        for (Kind kind : Kind.values()) {
            SegmentationRegions regions = SegmentationRegions.fromLabels(labels,kind,()->false);
            int expected = (kind.semantic ? 3 : 2) * (kind.volume ? 1 : 2) * 2;
            assertEquals(expected, regions.objects().size());
            Set<String> ids = new HashSet<>();
            for (SegmentationRegions.Region r : regions.objects()) {
                assertTrue(ids.add(r.id));
                assertEquals(kind.volume ? 2 : 1, r.slices().size());
            }
        }
        int[] left = {3,0,1,3,2,1}, bridge = {3,0,3};
        SegmentationRegions joined = SegmentationRegions.fromRuns(3,1,2,1,Kind.SEMANTIC_VOLUME,
                (z,t)->z == 0 ? left : bridge,()->false);
        assertEquals(1, joined.objects().size());
        assertEquals(2, joined.objects().get(0).slices().get(0).runCount());
    }

    @Test public void modesPublishOnlyRequestedFilesAndFailureIsPerImage() throws Exception {
        File dataset = tmp.newFolder("input"), source = new File(dataset,"image.tif");
        Files.write(source.toPath(), new byte[]{42});
        Tensor<IntType> empty = Tensor.build("labels","xy",ArrayImgs.ints(4,3));
        for (SegmentationOutput mode : SegmentationOutput.values()) {
            InferenceRunFiles run = InferenceRunFiles.create(source,"test","model",Collections.singletonList(source),null);
            mode.save(null, run, source, empty, Kind.INSTANCE_PLANES, ()->false, s->{});
            Set<String> files = new HashSet<>(Arrays.asList(run.getDirectory().list()));
            assertEquals(mode != SegmentationOutput.ROIS, files.remove("image_labels.tif"));
            assertEquals(mode != SegmentationOutput.LABELS, files.remove("image.geojson"));
            assertTrue(files.isEmpty());
        }
        InferenceRunFiles run = InferenceRunFiles.create(source,"test","model",Collections.singletonList(source),null);
        Tensor<IntType> bad = Tensor.build("bad","xy",ArrayImgs.ints(new int[]{1,-1},2,1));
        try {
            SegmentationOutput.BOTH.save(null,run,source,bad,Kind.INSTANCE_PLANES,()->false,s->{});
            fail("invalid labels");
        } catch (IOException expected) { assertTrue(expected.getMessage().contains("integer labels")); }
        assertEquals(0, run.getDirectory().list().length);
        assertArrayEquals(new byte[]{42}, Files.readAllBytes(source.toPath()));
    }

    @Test public void sparseUnsignedLabelIdsSurviveRoiAndTiffOutput() throws Exception {
        Tensor<net.imglib2.type.numeric.integer.UnsignedIntType> labels = Tensor.build("labels","xy",
                ArrayImgs.unsignedInts(new int[]{16777217,0,-1},3,1));
        SegmentationRegions regions = SegmentationRegions.fromLabels(labels,Kind.INSTANCE_PLANES,()->false);
        assertEquals(16777217L, regions.objects().get(0).label);
        assertEquals(4294967295L, regions.objects().get(1).label);
        File target = tmp.newFile("ids.tif");
        io.bioimage.modelrunner.gui.custom.LabelMaskWriter.write(labels,target);
        assertArrayEquals(new double[]{16777217,0,4294967295L},javax.imageio.ImageIO.read(target)
                .getRaster().getSamples(0,0,3,1,0,(double[])null),0);
    }
}
