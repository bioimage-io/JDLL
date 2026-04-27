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
package io.bioimage.modelrunner.model.detection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public final class Detection {

    private final String parentName;
    private final int batchIndex;
    private final double x1;
    private final double y1;
    private final double x2;
    private final double y2;
    private final double confidence;
    private final int classId;

    public Detection(String parentName, int batchIndex, double x1, double y1, double x2, double y2,
            double confidence, int classId) {
        this.parentName = parentName;
        this.batchIndex = batchIndex;
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.confidence = confidence;
        this.classId = classId;
    }

    public String getParentName() {
        return parentName;
    }

    public int getBatchIndex() {
        return batchIndex;
    }

    public double getX1() {
        return x1;
    }

    public double getY1() {
        return y1;
    }

    public double getX2() {
        return x2;
    }

    public double getY2() {
        return y2;
    }

    public double getConfidence() {
        return confidence;
    }

    public int getClassId() {
        return classId;
    }

    /**
     * Decodes a tensor containing detections in {@code [batch, detections, 6]}
     * layout into a flat list of detections.
     * <p>
     * The last dimension is expected to contain
     * {@code x1, y1, x2, y2, confidence, classId}. Rows where all six values are
     * zero are treated as padding and ignored. All detections keep
     * {@link Tensor#getName()} as their parent name.
     *
     * @param detectionsTensor tensor with shape {@code [batch, detections, 6]}
     * @param <T> tensor data type
     * @return flat list of detections, each carrying its batch index
     */
    public static <T extends RealType<T> & NativeType<T>>
    List<Detection> fromBN6Tensor(final Tensor<T> detectionsTensor) {
        if (detectionsTensor == null) {
            throw new IllegalArgumentException("Detections tensor cannot be null.");
        }
        final RandomAccessibleInterval<T> detections = detectionsTensor.getData();
        final long[] dims = detections.dimensionsAsLongArray();
        if (dims.length != 3 || dims[2] != 6) {
            throw new IllegalArgumentException("Expected detection tensor with shape [batch, detections, 6]. Got "
                    + Arrays.toString(dims) + ".");
        }

        final int batchSize = Math.toIntExact(dims[0]);
        final int nDetections = Math.toIntExact(dims[1]);
        final String parentName = detectionsTensor.getName();
        final RandomAccess<T> access = detections.randomAccess();
        final List<Detection> out = new ArrayList<Detection>();

        for (int b = 0; b < batchSize; b++) {
            for (int n = 0; n < nDetections; n++) {
                double x1 = getValue(access, b, n, 0);
                double y1 = getValue(access, b, n, 1);
                double x2 = getValue(access, b, n, 2);
                double y2 = getValue(access, b, n, 3);
                double confidence = getValue(access, b, n, 4);
                double classId = getValue(access, b, n, 5);
                if (isZeroDetection(x1, y1, x2, y2, confidence, classId)) {
                    continue;
                }
                out.add(new Detection(parentName, b, x1, y1, x2, y2, confidence, (int) Math.round(classId)));
            }
        }
        return out;
    }

    private static boolean isZeroDetection(double x1, double y1, double x2, double y2, double confidence, double classId) {
        return x1 == 0.0 && y1 == 0.0 && x2 == 0.0 && y2 == 0.0 && confidence == 0.0 && classId == 0.0;
    }

    private static <T extends RealType<T> & NativeType<T>>
    double getValue(final RandomAccess<T> access, final int batch, final int detection, final int field) {
        access.setPosition(batch, 0);
        access.setPosition(detection, 1);
        access.setPosition(field, 2);
        return access.get().getRealDouble();
    }
}
