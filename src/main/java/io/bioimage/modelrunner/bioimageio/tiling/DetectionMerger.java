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
package io.bioimage.modelrunner.bioimageio.tiling;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import io.bioimage.modelrunner.model.detection.Detection;

/**
 * Merges object detections produced from overlapping spatial tiles.
 * <p>
 * The merger shifts tile-local boxes back into full-image coordinates, clips
 * them to image bounds and removes duplicated detections with class-aware NMS.
 */
public final class DetectionMerger {
	
	private final long[] outputImageSize;
	
	private List<TileDetections> tileDetections = new ArrayList<TileDetections>();

    public static final double DEFAULT_NMS_IOU_THRESHOLD = 0.5d;

    public DetectionMerger(long[] outputSize) {
    	outputImageSize = outputSize;
    }
    
    public void addTileDetections(List<Detection> detections, long[] tilePos) {
    	tileDetections.add(new TileDetections(detections, tilePos));
    }

    public List<Detection> merge() {
        return merge(DEFAULT_NMS_IOU_THRESHOLD);
    }

    public List<Detection> merge(final double nmsIouThreshold) {
        if (tileDetections == null || tileDetections.isEmpty()) {
            return Collections.emptyList();
        }
        final List<Detection> shifted = new ArrayList<Detection>();
        for (TileDetections tile : tileDetections) {
            if (tile == null || tile.getDetections().isEmpty()) {
                continue;
            }
            for (Detection detection : tile.getDetections()) {
                Detection global = toGlobalDetection(detection, tile.getTilePosition());
                if (global != null) {
                    shifted.add(global);
                }
            }
        }
        return classAwareNms(shifted, nmsIouThreshold);
    }

    private Detection toGlobalDetection(final Detection detection, final long[] window) {
        if (detection == null || window == null || !isValid(detection)) {
            return null;
        }
        long w = this.outputImageSize[2] - this.outputImageSize[0];
        long h = this.outputImageSize[3] - this.outputImageSize[1];
        long tw = window[2] - window[0];
        long th = window[3] - window[1];
        double x1 = clip(detection.getX1() + tw, 0.0d, w);
        double y1 = clip(detection.getY1() + th, 0.0d, h);
        double x2 = clip(detection.getX2() + tw, 0.0d, w);
        double y2 = clip(detection.getY2() + th, 0.0d, h);
        if (x2 <= x1 || y2 <= y1) {
            return null;
        }
        return new Detection(detection.getParentName(), detection.getBatchIndex(), x1, y1, x2, y2,
                detection.getConfidence(), detection.getClassId());
    }

    private static List<Detection> classAwareNms(final List<Detection> detections,
            final double iouThreshold) {
        if (detections.isEmpty()) {
            return Collections.emptyList();
        }
        final double threshold = Double.isFinite(iouThreshold) && iouThreshold >= 0.0d
                ? iouThreshold
                : DEFAULT_NMS_IOU_THRESHOLD;
        final List<Detection> sorted = new ArrayList<Detection>(detections);
        sorted.sort(Comparator.comparingDouble(Detection::getConfidence).reversed());

        final List<Detection> kept = new ArrayList<Detection>();
        for (Detection candidate : sorted) {
            boolean duplicate = false;
            for (Detection selected : kept) {
                if (sameNmsGroup(candidate, selected) && iou(candidate, selected) > threshold) {
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
        double x1 = Math.max(a.getX1(), b.getX1());
        double y1 = Math.max(a.getY1(), b.getY1());
        double x2 = Math.min(a.getX2(), b.getX2());
        double y2 = Math.min(a.getY2(), b.getY2());
        double intersection = area(x1, y1, x2, y2);
        if (intersection <= 0.0d) {
            return 0.0d;
        }
        double union = area(a.getX1(), a.getY1(), a.getX2(), a.getY2())
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

    private static final class TileDetections {

        private final long[] tileWindow;
        private final List<Detection> detections;

        public TileDetections(final List<Detection> detections, long[] tileWindow) {
            if (tileWindow == null) {
                throw new IllegalArgumentException("Tile window cannot be null.");
            }
            this.tileWindow = tileWindow;
            this.detections = detections == null
                    ? Collections.<Detection>emptyList()
                    : Collections.unmodifiableList(new ArrayList<Detection>(detections));
        }

        public long[] getTilePosition() {
            return tileWindow;
        }

        public List<Detection> getDetections() {
            return detections;
        }
    }
}
