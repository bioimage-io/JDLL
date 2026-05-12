/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.model.tiling.merger;

import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.bioimage.modelrunner.model.detection.Detection;
import io.bioimage.modelrunner.model.tiling.TileInfo;
import io.bioimage.modelrunner.model.tiling.TileMaker;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

/**
 * Reconstructs object detections produced from spatial tiles.
 * <p>
 * The raw model outputs are expected to be tensors with axes {@code bic}
 * ({@code batch, instance, columns}). Non-{@code bic} outputs are ignored. The
 * decoded detections are shifted from tile-local coordinates into the reference
 * image coordinate system, clipped to image bounds and filtered with class-aware
 * NMS.
 */
public final class DetectionMerger<T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
        extends Merger<Tensor<T>, Detection> {

    public static final double DEFAULT_NMS_IOU_THRESHOLD = 0.5d;
    public static final double DEFAULT_MIN_OBJECT_IMAGE_AREA_RATIO = 0.001d;
    public static final double DEFAULT_TARGET_OBJECT_TILE_AREA_RATIO = 0.01d;

    private enum Mode {
        TILE_MAKER,
        PATCH_SIZES,
        OBJECT_SIZE
    }

    private final Mode mode;
    private final TileMaker tileMaker;
    private final List<TileInfo> patchSizes;
    private final Rectangle objectRectangle;
    private long[] sizeArray;

    private float tileOverlap = 0.15f;
    private double nmsIouThreshold = DEFAULT_NMS_IOU_THRESHOLD;

    private List<Tensor<T>> inputs = Collections.emptyList();
    private List<InputImage> imageInputs = Collections.emptyList();
    private List<long[]> referenceWindows = Collections.emptyList();
    private long[] referenceWindow;
    private List<List<Detection>> detectionsByPatch = Collections.emptyList();
    private List<Detection> reconstructed = Collections.emptyList();
    private boolean reconstructedValid;

    public DetectionMerger(final TileMaker tileMaker) {
        if (tileMaker == null) {
            throw new IllegalArgumentException("TileMaker cannot be null.");
        }
        this.mode = Mode.TILE_MAKER;
        this.tileMaker = tileMaker;
        this.patchSizes = Collections.emptyList();
        this.objectRectangle = null;
    }

    public DetectionMerger(final List<TileInfo> patchSizes) {
        if (patchSizes == null || patchSizes.isEmpty()) {
            throw new IllegalArgumentException("Patch sizes cannot be null or empty.");
        }
        this.mode = Mode.PATCH_SIZES;
        this.tileMaker = null;
        this.patchSizes = Collections.unmodifiableList(new ArrayList<TileInfo>(patchSizes));
        this.objectRectangle = null;
    }

    public DetectionMerger(final Rectangle objectRectangle) {
        this.mode = Mode.OBJECT_SIZE;
        this.tileMaker = null;
        this.patchSizes = Collections.emptyList();
        this.objectRectangle = objectRectangle;
    }

    public DetectionMerger<T, R> setTileOverlap(final float tileOverlap) {
        if (!Float.isFinite(tileOverlap) || tileOverlap < 0.0f || tileOverlap >= 1.0f) {
            throw new IllegalArgumentException("Tile overlap must be >= 0 and < 1.");
        }
        this.tileOverlap = tileOverlap;
        reconfigureIfNeeded();
        return this;
    }

    public DetectionMerger<T, R> setNmsIouThreshold(final double nmsIouThreshold) {
        if (!Double.isFinite(nmsIouThreshold) || nmsIouThreshold < 0.0d) {
            throw new IllegalArgumentException("NMS IoU threshold must be finite and >= 0.");
        }
        this.nmsIouThreshold = nmsIouThreshold;
        this.reconstructedValid = false;
        return this;
    }

    public DetectionMerger<T, R> setNmsThreshold(final double nmsIouThreshold) {
        return setNmsIouThreshold(nmsIouThreshold);
    }

    @Override
    public void configure(final List<Tensor<T>> inputs) {
        this.inputs = inputs == null ? Collections.<Tensor<T>>emptyList() : inputs;
        this.imageInputs = findImageInputs(this.inputs);
        if (imageInputs.isEmpty()) {
            throw new IllegalArgumentException("DetectionMerger needs at least one input tensor with x and y axes.");
        }
        final InputImage reference = imageInputs.get(0);
        this.referenceWindow = new long[] { 0, 0, reference.width(), reference.height() };
        this.referenceWindows = createReferenceWindows(reference);
        this.detectionsByPatch = new ArrayList<List<Detection>>(referenceWindows.size());
        for (int i = 0; i < referenceWindows.size(); i++) {
            detectionsByPatch.add(null);
        }
        this.reconstructed = Collections.emptyList();
        this.reconstructedValid = false;
        this.configured = true;
        this.digested = referenceWindows.isEmpty();
    }

    @Override
    public List<Tensor<T>> get(final int patchNumber) {
        requireConfigured();
        patchNumberValid(patchNumber);
        if (mode == Mode.TILE_MAKER) {
            return getTileMakerPatch(patchNumber);
        }
        final long[] referenceTileWindow = referenceWindows.get(patchNumber);
        final List<Tensor<T>> patch = new ArrayList<Tensor<T>>(inputs.size());
        for (Tensor<T> input : inputs) {
            final InputImage imageInput = findImageInput(input);
            if (imageInput == null) {
                patch.add(input);
            } else {
                patch.add(extractPatch(imageInput, scaleWindow(referenceTileWindow, imageInputs.get(0), imageInput)));
            }
        }
        return patch;
    }

    @Override
    public int getNPatches() {
        requireConfigured();
        return referenceWindows.size();
    }

    @Override
    public void digest(final int patchNumber, final List<Detection> outputs) {
        requireConfigured();
        patchNumberValid(patchNumber);
        detectionsByPatch.set(patchNumber, immutableCopy(outputs));
        reconstructed = Collections.emptyList();
        reconstructedValid = false;
        digested = allPatchesDigested();
    }

    public void digestTensors(final int patchNumber, final List<Tensor<R>> outputs) {
        digest(patchNumber, decodeBicOutputs(outputs));
    }

    @Override
    public List<Detection> getReconstructed() {
        requireConfigured();
        requireDigested();
        if (reconstructedValid) {
            return reconstructed;
        }
        final List<Detection> shifted = new ArrayList<Detection>();
        for (int i = 0; i < detectionsByPatch.size(); i++) {
            final List<Detection> detections = detectionsByPatch.get(i);
            if (detections == null || detections.isEmpty()) {
                continue;
            }
            for (Detection detection : detections) {
                final Detection global = toGlobalDetection(detection, referenceWindows.get(i));
                if (global != null) {
                    shifted.add(global);
                }
            }
        }
        reconstructed = immutableCopy(classAwareNms(shifted, nmsIouThreshold));
        reconstructedValid = true;
        return reconstructed;
    }

    private List<Tensor<T>> getTileMakerPatch(final int patchNumber) {
        final List<Tensor<T>> patch = new ArrayList<Tensor<T>>(inputs.size());
        for (Tensor<T> input : inputs) {
            if (hasXY(input)) {
                patch.add(tileMaker.getNthTileInput(input, patchNumber));
            } else {
                patch.add(input);
            }
        }
        return patch;
    }

    private List<long[]> createReferenceWindows(final InputImage reference) {
        if (mode == Mode.TILE_MAKER) {
            return createTileMakerWindows(reference);
        } else if (mode == Mode.PATCH_SIZES) {
            return createPatchSizeWindows(reference);
        }
        if (this.objectRectangle == null) {
        	sizeArray = new long[] {0, 0, reference.width(), reference.height()};
        } else {
        	sizeArray = new long[] {objectRectangle.x, objectRectangle.y,
        			objectRectangle.width + objectRectangle.x, objectRectangle.y + objectRectangle.height};
        }
        return createObjectSizeWindows(reference);
    }

    private List<long[]> createTileMakerWindows(final InputImage reference) {
        final List<long[]> positions = tileMaker.getTilePostionsInputImage(reference.tensor.getName());
        final long[] tileSize = tileMaker.getInputTileSize(reference.tensor.getName());
        final List<long[]> windows = new ArrayList<long[]>(positions.size());
        for (long[] position : positions) {
            windows.add(toXyxyWindow(position, tileSize, reference.axes));
        }
        return Collections.unmodifiableList(windows);
    }

    private List<long[]> createPatchSizeWindows(final InputImage reference) {
        validatePatchSizes();
        final TileInfo referencePatch = patchSizeFor(reference.tensor.getName());
        final long[] tileSize = referencePatch.getTileDims();
        final String tileAxes = referencePatch.getTileAxesOrder();
        return createWindows(reference.width(), reference.height(), axisSize(tileSize, tileAxes, 'x'),
                axisSize(tileSize, tileAxes, 'y'));
    }

    private List<long[]> createObjectSizeWindows(final InputImage reference) {
        final long objectWidth = Math.max(1L, sizeArray[X2] - sizeArray[X1]);
        final long objectHeight = Math.max(1L, sizeArray[Y2] - sizeArray[Y1]);
        final double objectArea = objectWidth * (double) objectHeight;
        final double imageArea = reference.width() * (double) reference.height();
        if (objectArea / imageArea >= DEFAULT_MIN_OBJECT_IMAGE_AREA_RATIO) {
            return Collections.singletonList(referenceWindow.clone());
        }
        final double scale = Math.sqrt(1.0d / DEFAULT_TARGET_OBJECT_TILE_AREA_RATIO);
        final long tileWidth = Math.min(reference.width(), Math.max(1L, Math.round(objectWidth * scale)));
        final long tileHeight = Math.min(reference.height(), Math.max(1L, Math.round(objectHeight * scale)));
        return createWindows(reference.width(), reference.height(), tileWidth, tileHeight);
    }

    private List<long[]> createWindows(final long imageWidth, final long imageHeight,
            final long requestedTileWidth, final long requestedTileHeight) {
        final long tileWidth = Math.min(imageWidth, Math.max(1L, requestedTileWidth));
        final long tileHeight = Math.min(imageHeight, Math.max(1L, requestedTileHeight));
        if (tileWidth >= imageWidth && tileHeight >= imageHeight) {
            return Collections.singletonList(new long[] { 0, 0, imageWidth, imageHeight });
        }
        final List<Long> xs = starts(imageWidth, tileWidth);
        final List<Long> ys = starts(imageHeight, tileHeight);
        final List<long[]> windows = new ArrayList<long[]>(xs.size() * ys.size());
        for (long y : ys) {
            for (long x : xs) {
                windows.add(new long[] { x, y, x + tileWidth, y + tileHeight });
            }
        }
        return Collections.unmodifiableList(windows);
    }

    private List<Long> starts(final long imageSize, final long tileSize) {
        if (tileSize >= imageSize) {
            return Collections.singletonList(0L);
        }
        final long step = Math.max(1L, Math.round(tileSize * (1.0d - tileOverlap)));
        final List<Long> starts = new ArrayList<Long>();
        long start = 0L;
        while (start + tileSize < imageSize) {
            starts.add(start);
            start += step;
        }
        final long last = imageSize - tileSize;
        if (starts.isEmpty() || starts.get(starts.size() - 1).longValue() != last) {
            starts.add(last);
        }
        return starts;
    }

    private void validatePatchSizes() {
        if (patchSizes.size() != imageInputs.size()) {
            throw new IllegalArgumentException("Expected one TileInfo per image input. Found " + patchSizes.size()
                    + " TileInfo objects for " + imageInputs.size() + " image inputs.");
        }
        final Map<String, TileInfo> byName = patchSizeMap();
        for (InputImage imageInput : imageInputs) {
            final TileInfo info = byName.get(imageInput.tensor.getName());
            if (info == null) {
                throw new IllegalArgumentException("Missing TileInfo for image tensor '"
                        + imageInput.tensor.getName() + "'.");
            }
            validateTileInfoMatchesInput(info, imageInput);
        }
    }

    private Map<String, TileInfo> patchSizeMap() {
        final Map<String, TileInfo> byName = new HashMap<String, TileInfo>();
        for (TileInfo info : patchSizes) {
            byName.put(info.getName(), info);
        }
        return byName;
    }

    private TileInfo patchSizeFor(final String tensorName) {
        final TileInfo info = patchSizeMap().get(tensorName);
        if (info == null) {
            throw new IllegalArgumentException("Missing TileInfo for image tensor '" + tensorName + "'.");
        }
        return info;
    }

    private void validateTileInfoMatchesInput(final TileInfo info, final InputImage input) {
        if (axisSize(info.getImageDims(), info.getImageAxesOrder(), 'x') != input.width()
                || axisSize(info.getImageDims(), info.getImageAxesOrder(), 'y') != input.height()) {
            throw new IllegalArgumentException("TileInfo image size for '" + info.getName()
                    + "' does not match the input tensor size.");
        }
        axisSize(info.getTileDims(), info.getTileAxesOrder(), 'x');
        axisSize(info.getTileDims(), info.getTileAxesOrder(), 'y');
    }

    private List<Detection> decodeBicOutputs(final List<Tensor<R>> outputs) {
        if (outputs == null || outputs.isEmpty()) {
            return Collections.emptyList();
        }
        final List<Detection> detections = new ArrayList<Detection>();
        for (Tensor<R> output : outputs) {
            if (output == null || !"bic".equals(output.getAxesOrderString())) {
                continue;
            }
            detections.addAll(Detection.fromBN6Tensor(output));
        }
        return immutableCopy(detections);
    }

    private Detection toGlobalDetection(final Detection detection, final long[] window) {
        if (detection == null || !isValid(detection)) {
            return null;
        }
        validateWindow(window, "Tile window");
        final double x1 = clip(detection.getX1() + window[X1], 0.0d, referenceWindow[X2]);
        final double y1 = clip(detection.getY1() + window[Y1], 0.0d, referenceWindow[Y2]);
        final double x2 = clip(detection.getX2() + window[X1], 0.0d, referenceWindow[X2]);
        final double y2 = clip(detection.getY2() + window[Y1], 0.0d, referenceWindow[Y2]);
        if (x2 <= x1 || y2 <= y1) {
            return null;
        }
        return new Detection(detection.getParentName(), detection.getBatchIndex(), x1, y1, x2, y2,
                detection.getConfidence(), detection.getClassId());
    }

    private Tensor<T> extractPatch(final InputImage imageInput, final long[] window) {
        final RandomAccessibleInterval<T> data = imageInput.tensor.getData();
        final long[] min = new long[data.numDimensions()];
        final long[] max = new long[data.numDimensions()];
        final long[] dims = data.dimensionsAsLongArray();
        for (int d = 0; d < dims.length; d++) {
            min[d] = 0L;
            max[d] = dims[d] - 1L;
        }
        min[imageInput.xAxis] = window[X1];
        min[imageInput.yAxis] = window[Y1];
        max[imageInput.xAxis] = window[X2] - 1L;
        max[imageInput.yAxis] = window[Y2] - 1L;
        final RandomAccessibleInterval<T> patch = Views.interval(
                Views.extendMirrorDouble(data), new FinalInterval(min, max));
        return Tensor.build(imageInput.tensor.getName(), imageInput.axes, patch);
    }

    private InputImage findImageInput(final Tensor<T> tensor) {
        for (InputImage input : imageInputs) {
            if (input.tensor == tensor) {
                return input;
            }
        }
        return null;
    }

    private List<InputImage> findImageInputs(final List<Tensor<T>> inputs) {
        if (inputs == null || inputs.isEmpty()) {
            return Collections.emptyList();
        }
        final List<InputImage> imageInputs = new ArrayList<InputImage>();
        for (Tensor<T> input : inputs) {
            if (hasXY(input)) {
                imageInputs.add(new InputImage(input));
            }
        }
        return Collections.unmodifiableList(imageInputs);
    }

    private static <T extends RealType<T> & NativeType<T>> boolean hasXY(final Tensor<T> tensor) {
        return tensor != null
                && tensor.getAxesOrderString().indexOf('x') >= 0
                && tensor.getAxesOrderString().indexOf('y') >= 0;
    }

    private long[] scaleWindow(final long[] referenceWindow, final InputImage reference,
            final InputImage target) {
        if (reference == target) {
            return referenceWindow.clone();
        }
        return new long[] {
                scale(referenceWindow[X1], reference.width(), target.width()),
                scale(referenceWindow[Y1], reference.height(), target.height()),
                scale(referenceWindow[X2], reference.width(), target.width()),
                scale(referenceWindow[Y2], reference.height(), target.height())
        };
    }

    private static long scale(final long value, final long sourceSize, final long targetSize) {
        return clip(Math.round(value * (targetSize / (double) sourceSize)), 0L, targetSize);
    }

    private static long clip(final long value, final long min, final long max) {
        return Math.max(min, Math.min(max, value));
    }

    private static long axisSize(final long[] dims, final String axes, final char axis) {
        if (dims == null || axes == null) {
            throw new IllegalArgumentException("Dimensions and axes cannot be null.");
        }
        final int index = axes.indexOf(axis);
        if (index < 0 || index >= dims.length) {
            throw new IllegalArgumentException("Axes '" + axes + "' do not contain axis '" + axis + "'.");
        }
        return dims[index];
    }

    private static long[] toXyxyWindow(final long[] tilePosition, final long[] tileSize, final String axes) {
        final int x = axes.indexOf('x');
        final int y = axes.indexOf('y');
        if (x < 0 || y < 0 || tilePosition == null || tileSize == null
                || tilePosition.length <= Math.max(x, y) || tileSize.length <= Math.max(x, y)) {
            throw new IllegalArgumentException("Tile position and size must contain x and y dimensions.");
        }
        return new long[] {
                tilePosition[x],
                tilePosition[y],
                tilePosition[x] + tileSize[x],
                tilePosition[y] + tileSize[y]
        };
    }

    private static long[] toWindow(final Rectangle rectangle) {
        if (rectangle == null) {
            throw new IllegalArgumentException("Object rectangle cannot be null.");
        }
        return new long[] {
                rectangle.x,
                rectangle.y,
                rectangle.x + rectangle.width,
                rectangle.y + rectangle.height
        };
    }

    private static List<Detection> classAwareNms(final List<Detection> detections,
            final double iouThreshold) {
        if (detections.isEmpty()) {
            return Collections.emptyList();
        }
        final List<Detection> sorted = new ArrayList<Detection>(detections);
        sorted.sort(Comparator.comparingDouble(Detection::getConfidence).reversed());

        final List<Detection> kept = new ArrayList<Detection>();
        for (Detection candidate : sorted) {
            boolean duplicate = false;
            for (Detection selected : kept) {
                if (sameNmsGroup(candidate, selected) && iou(candidate, selected) > iouThreshold) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                kept.add(candidate);
            }
        }
        return kept;
    }

    private static boolean sameNmsGroup(final Detection a, final Detection b) {
        return a.getBatchIndex() == b.getBatchIndex() && a.getClassId() == b.getClassId();
    }

    private static double iou(final Detection a, final Detection b) {
        final double x1 = Math.max(a.getX1(), b.getX1());
        final double y1 = Math.max(a.getY1(), b.getY1());
        final double x2 = Math.min(a.getX2(), b.getX2());
        final double y2 = Math.min(a.getY2(), b.getY2());
        final double intersection = area(x1, y1, x2, y2);
        if (intersection <= 0.0d) {
            return 0.0d;
        }
        final double union = area(a.getX1(), a.getY1(), a.getX2(), a.getY2())
                + area(b.getX1(), b.getY1(), b.getX2(), b.getY2())
                - intersection;
        return union <= 0.0d ? 0.0d : intersection / union;
    }

    private static double area(final double x1, final double y1, final double x2, final double y2) {
        return Math.max(0.0d, x2 - x1) * Math.max(0.0d, y2 - y1);
    }

    private static boolean isValid(final Detection detection) {
        return Double.isFinite(detection.getX1())
                && Double.isFinite(detection.getY1())
                && Double.isFinite(detection.getX2())
                && Double.isFinite(detection.getY2())
                && Double.isFinite(detection.getConfidence())
                && detection.getX2() > detection.getX1()
                && detection.getY2() > detection.getY1();
    }

    private static double clip(final double value, final double min, final double max) {
        return Math.max(min, Math.min(max, value));
    }

    private boolean allPatchesDigested() {
        for (List<Detection> detections : detectionsByPatch) {
            if (detections == null) {
                return false;
            }
        }
        return true;
    }

    private void reconfigureIfNeeded() {
        if (configured) {
            configure(inputs);
        }
    }

    private final class InputImage {

        private final Tensor<T> tensor;
        private final String axes;
        private final int xAxis;
        private final int yAxis;
        private final long[] dims;

        private InputImage(final Tensor<T> tensor) {
            this.tensor = tensor;
            this.axes = tensor.getAxesOrderString();
            this.xAxis = axes.indexOf('x');
            this.yAxis = axes.indexOf('y');
            this.dims = tensor.getData().dimensionsAsLongArray();
        }

        private long width() {
            return dims[xAxis];
        }

        private long height() {
            return dims[yAxis];
        }
    }
}
