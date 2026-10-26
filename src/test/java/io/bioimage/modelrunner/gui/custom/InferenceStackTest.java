/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.gui.custom;

import static org.junit.Assert.*;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.gui.custom.yolo.YoloDetectionCsvWriter;
import io.bioimage.modelrunner.gui.custom.yolo.YoloDetectionGeoJsonWriter;
import io.bioimage.modelrunner.gui.custom.yolo.YoloInferenceService;
import io.bioimage.modelrunner.model.InferenceProgress;
import io.bioimage.modelrunner.model.detection.Detection;
import io.bioimage.modelrunner.model.special.yolo.Yolo;
import io.bioimage.modelrunner.model.tiling.merger.Merger;
import io.bioimage.modelrunner.model.tiling.merger.NoTileMerger;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

public class InferenceStackTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test
    public void planarInferenceKeepsEveryZAndTimeAndAggregatesProgress() throws Exception {
        RandomAccessibleInterval<FloatType> image = source();
        InferenceStack<FloatType> stack = new InferenceStack<>(Views.translate(image, 4, 5, 6, 7, 8), false);
        List<InferenceProgress> events = new ArrayList<>();
        Consumer<InferenceProgress> progress = stack.progress(events::add);
        List<Tensor<FloatType>> result = stack.run(plane -> {
            assertArrayEquals(new long[] {3, 2, 1}, plane.dimensionsAsLongArray());
            progress.accept(InferenceProgress.inferenceStart(2));
            for (int patch = 1; patch <= 2; patch++) progress.accept(InferenceProgress.patchEnd(patch, 2));
            progress.accept(InferenceProgress.inferenceEnd());
            return Collections.singletonList(Tensor.build("labels", "xy", Views.hyperSlice(plane, 2, 0)));
        }, () -> false);
        Tensor<FloatType> output = result.get(0);
        assertEquals("xyzt", output.getAxesOrderString());
        assertArrayEquals(new long[] {3, 2, 2, 3}, output.getData().dimensionsAsLongArray());
        for (int t = 0; t < 3; t++) for (int z = 0; z < 2; z++)
            assertEquals(100 * t + 10 * z + 2, pixel(output.getData(), 2, 0, z, t), 0);
        assertEquals(InferenceProgress.Phase.INFERENCE_START, events.get(0).getPhase());
        assertEquals(12, events.get(0).getTotalPatches());
        for (int i = 1; i <= 12; i++) assertEquals(i, events.get(i).getPatchIndex());
        assertEquals(InferenceProgress.Phase.INFERENCE_END, events.get(13).getPhase());
    }

    @Test
    public void volumeInferenceKeepsCompleteZForEveryTimePoint() throws Exception {
        InferenceStack<FloatType> stack = new InferenceStack<>(source(), true);
        AtomicInteger calls = new AtomicInteger();
        List<Tensor<FloatType>> result = stack.run(volume -> {
            calls.incrementAndGet();
            assertArrayEquals(new long[] {3, 2, 1, 2}, volume.dimensionsAsLongArray());
            return Collections.singletonList(Tensor.build("labels", "xyz", Views.hyperSlice(volume, 2, 0)));
        }, () -> false);
        assertEquals(3, calls.get());
        Tensor<FloatType> output = result.get(0);
        assertEquals("xyzt", output.getAxesOrderString());
        assertArrayEquals(new long[] {3, 2, 2, 3}, output.getData().dimensionsAsLongArray());
        assertEquals(212, pixel(output.getData(), 2, 0, 1, 2), 0);
        RandomAccessibleInterval<FloatType> xyz = ArrayImgs.floats(3, 2, 4);
        assertArrayEquals(new long[] {3, 2, 1, 4}, new InferenceStack<>(xyz, true).input(0).dimensionsAsLongArray());
    }

    @Test
    public void singletonAxesAndSinglePlanesRetainTheirMeaning() throws Exception {
        for (long[] shape : new long[][] {{3, 2}, {3, 2, 1}, {3, 2, 1, 1}, {3, 2, 1, 1, 1}}) {
            InferenceStack<FloatType> stack = new InferenceStack<>(ArrayImgs.floats(shape), false);
            List<Tensor<FloatType>> result = stack.run(plane -> Collections.singletonList(
                    Tensor.build("labels", "xy", Views.hyperSlice(plane, 2, 0))), () -> false);
            assertEquals(shape.length == 5 ? "xyzt" : shape.length == 4 ? "xyz" : "xy",
                    result.get(0).getAxesOrderString());
        }
    }

    @Test
    public void cancellationBetweenEntriesDoesNotStartAnotherInference() {
        InferenceStack<FloatType> stack = new InferenceStack<>(source(), false);
        AtomicBoolean cancelled = new AtomicBoolean();
        AtomicInteger calls = new AtomicInteger();
        assertThrows(RunModelException.class, () -> stack.run(plane -> {
            calls.incrementAndGet();
            cancelled.set(true);
            return Collections.singletonList(Tensor.build("labels", "xy", Views.hyperSlice(plane, 2, 0)));
        }, cancelled::get));
        assertEquals(1, calls.get());
    }

    @Test
    public void tiffExportPreservesAllPlanesTimeMetadataAndLargeLabels() throws Exception {
        RandomAccessibleInterval<FloatType> labels = Views.permute(Views.hyperSlice(source(), 2, 0), 0, 1);
        for (FloatType pixel : Views.iterable(labels)) pixel.add(new FloatType(70000));
        File output = LabelMaskWriter.write(Tensor.build("labels", "yxzt", labels),
                new File(temporary.getRoot(), "labels.png"));
        assertEquals("labels.tif", output.getName());
        ImageReader reader = ImageIO.getImageReadersByFormatName("TIFF").next();
        try (ImageInputStream input = ImageIO.createImageInputStream(output)) {
            reader.setInput(input);
            assertEquals(6, reader.getNumImages(true));
            for (int t = 0; t < 3; t++) for (int z = 0; z < 2; z++)
                assertEquals(70000 + 100 * t + 10 * z + 2,
                        reader.read(t * 2 + z).getRaster().getSampleFloat(2, 0, 0), 0);
            org.w3c.dom.Node tree = reader.getImageMetadata(0).getAsTree("javax_imageio_tiff_image_1.0");
            String description = findDescription(tree);
            assertTrue(description.contains("slices=2\nframes=3"));
        } finally { reader.dispose(); }
    }

    @Test
    public void detectionsAndExportsRetainExplicitZTimeAndClassNames() throws Exception {
        Detection source = new Detection("image", 0, 1, 2, 3, 4, 0.9, 2);
        HashMap<String, String> names = new HashMap<>();
        names.put("2", "nucleus");
        source.setMappings(names);
        Detection positioned = source.withPosition(3, 4, 23);
        assertEquals("nucleus", positioned.getClassName());
        assertEquals(23, positioned.getBatchIndex());
        assertEquals(3, positioned.getZIndex());
        assertEquals(4, positioned.getTimeIndex());
        File csv = temporary.newFile("labels.csv"), json = temporary.newFile("labels.geojson");
        YoloDetectionCsvWriter.write(csv, Collections.singletonMap("image", Collections.singletonList(positioned)));
        YoloDetectionGeoJsonWriter.write(json, "image", Collections.singletonList(positioned));
        String csvText = new String(Files.readAllBytes(csv.toPath()), StandardCharsets.UTF_8);
        assertTrue(csvText.contains("confidence,z,t"));
        assertTrue(csvText.trim().endsWith(",3,4"));
        String jsonText = new String(Files.readAllBytes(json.toPath()), StandardCharsets.UTF_8);
        assertTrue(jsonText.contains("\"z\":3,\"t\":4"));
    }

    @Test
    public void yoloServiceRunsEveryPlaneAndPositionsTheDetections() throws Exception {
        FakeYolo model = new FakeYolo(temporary.newFile("fake.pt"));
        YoloInferenceService service = new YoloInferenceService(null);
        java.lang.reflect.Field field = YoloInferenceService.class.getDeclaredField("model");
        field.setAccessible(true);
        field.set(service, model);
        List<Detection> detections = service.runWithProgress("fake", source(), null, "cpu");
        assertEquals(6, detections.size());
        for (int unit = 0; unit < detections.size(); unit++) {
            Detection detection = detections.get(unit);
            assertEquals(unit % 2, detection.getZIndex());
            assertEquals(unit / 2, detection.getTimeIndex());
            assertEquals(unit, detection.getBatchIndex());
            assertEquals("nucleus", detection.getClassName());
            assertEquals(100 * (unit / 2) + 10 * (unit % 2), model.pixels.get(unit), 0);
        }
    }

    private static final class FakeYolo extends Yolo {
        private final List<Float> pixels = new ArrayList<>();
        FakeYolo(File weights) throws Exception { super(weights.getAbsolutePath(), "cpu"); loaded = true; }
        @Override public HashMap<String, String> getClassMappings() {
            HashMap<String, String> names = new HashMap<>();
            names.put("1", "nucleus");
            return names;
        }
        @Override protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
        Merger<Tensor<T>, Tensor<R>> getTileMaker(List<Tensor<T>> inputs) {
            NoTileMerger<T, R> merger = new NoTileMerger<>();
            merger.configure(inputs);
            return merger;
        }
        @Override @SuppressWarnings({"unchecked", "rawtypes"})
        protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
        List<Tensor<R>> backboneSingleInferenceTile(List<Tensor<T>> inputs) {
            pixels.add(inputs.get(0).getData().randomAccess().get().getRealFloat());
            return (List) Collections.singletonList(Tensor.build("detections", "bic",
                    ArrayImgs.floats(new float[] {0, 0, 2, 2, 0.9f, 1}, 1, 1, 6)));
        }
    }

    private static RandomAccessibleInterval<FloatType> source() {
        RandomAccessibleInterval<FloatType> image = ArrayImgs.floats(3, 2, 1, 2, 3);
        net.imglib2.RandomAccess<FloatType> access = image.randomAccess();
        for (int t = 0; t < 3; t++) for (int z = 0; z < 2; z++)
            for (int y = 0; y < 2; y++) for (int x = 0; x < 3; x++) {
                access.setPosition(new long[] {x, y, 0, z, t});
                access.get().set(100 * t + 10 * z + x);
            }
        return image;
    }

    private static float pixel(RandomAccessibleInterval<FloatType> image, long... position) {
        net.imglib2.RandomAccess<FloatType> access = image.randomAccess();
        access.setPosition(position);
        return access.get().get();
    }

    private static String findDescription(org.w3c.dom.Node node) {
        if ("TIFFAscii".equals(node.getNodeName())) return node.getAttributes().getNamedItem("value").getNodeValue();
        for (org.w3c.dom.Node child = node.getFirstChild(); child != null; child = child.getNextSibling()) {
            String found = findDescription(child);
            if (found != null) return found;
        }
        return null;
    }
}
