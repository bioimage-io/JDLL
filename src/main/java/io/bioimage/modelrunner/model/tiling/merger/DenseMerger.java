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
import java.util.List;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.model.detection.Detection;
import io.bioimage.modelrunner.model.tiling.TileInfo;
import io.bioimage.modelrunner.model.tiling.TileMaker;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Merges object detections produced from overlapping spatial tiles.
 * <p>
 * The merger shifts tile-local boxes back into full-image coordinates, clips
 * them to image bounds and removes duplicated detections with class-aware NMS.
 */
public final class DenseMerger<T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
extends Merger<Tensor<T>, Tensor<R>> {
	
	private final TileMaker tileMaker;
    private long[] sizeArray;

    private float tileOverlap = 0.15f;

    private List<Tensor<T>> inputs = Collections.emptyList();
    private List<InputImage> imageInputs = Collections.emptyList();
    private List<long[]> referenceWindows = Collections.emptyList();
    private long[] referenceWindow;
    private List<Tensor<R>> reconstructed = Collections.emptyList();
    private Tensor<R> outputPrototype;
    private boolean reconstructedValid;

    private static final int X1 = 0;
    private static final int Y1 = 1;
    private static final int X2 = 2;
    private static final int Y2 = 3;
    private static final int WINDOW_LENGTH = 4;


    public static final double DEFAULT_NMS_IOU_THRESHOLD = 0.5d;

    public DenseMerger(final TileMaker tileMaker) {
        if (tileMaker == null) {
            throw new IllegalArgumentException("TileMaker cannot be null.");
        }
        this.tileMaker = tileMaker;
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

        this.reconstructed = tileMaker.createOutputTensors((R) new net.imglib2.type.numeric.real.FloatType(0.0f));
        this.outputPrototype = null;
        this.reconstructedValid = false;
        this.configured = true;
        this.digested = referenceWindows.isEmpty();
    }

	@Override
	public List<Tensor<T>> get(int patchNumber) {
        requireConfigured();
        patchNumberValid(patchNumber);
        return getTileMakerPatch(patchNumber);
	}

	@Override
	public int getNPatches() {
        requireConfigured();
        return referenceWindows.size();
	}

    @Override
    public void digest(final int patchNumber, final List<Tensor<R>> outputs) {
        requireConfigured();
        patchNumberValid(patchNumber);
        for (int i = 0; i < outputs.size(); i ++) {
        	tileMaker.getNthTileOutput(reconstructed.get(0), patchNumber) = outputs.get(i);
        }
                
        reconstructed = Collections.emptyList();
        reconstructedValid = false;
        digested = patchNumber + 1 == this.getNPatches();
    }

    @Override
    public List<Tensor<R>> getReconstructed() {
        requireConfigured();
        requireDigested();
        return reconstructed;
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

    private List<long[]> createReferenceWindows(final InputImage reference) {
        return createTileMakerWindows(reference);
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

    private static <T extends RealType<T> & NativeType<T>> boolean hasXY(final Tensor<T> tensor) {
        return tensor != null
                && tensor.getAxesOrderString().indexOf('x') >= 0
                && tensor.getAxesOrderString().indexOf('y') >= 0;
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
