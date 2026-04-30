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
package io.bioimage.modelrunner.model.special.yolo;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

public final class YoloTrainingProgress {

    public static final String TRAIN_BOX_LOSS = "train/box_loss";
    public static final String TRAIN_CLS_LOSS = "train/cls_loss";
    public static final String TRAIN_DFL_LOSS = "train/dfl_loss";
    public static final String VAL_BOX_LOSS = "val/box_loss";
    public static final String VAL_CLS_LOSS = "val/cls_loss";
    public static final String VAL_DFL_LOSS = "val/dfl_loss";
    public static final String MAP50_95 = "metrics/mAP50-95(B)";
    public static final String MAP50 = "metrics/mAP50(B)";
    public static final String PRECISION = "metrics/precision(B)";
    public static final String RECALL = "metrics/recall(B)";
    public static final String YOLO_TOTAL_LOSS_LABEL =
            "YOLO loss = box_loss + cls_loss + dfl_loss";
    public static final String PRIMARY_DETECTION_METRIC_LABEL = "mAP50-95(B)";

    private final int epoch;
    private final int step;
    private final int totalEpochs;
    private final int totalSteps;
    private final Map<String, Double> losses;
    private final Map<String, Double> metrics;

    public YoloTrainingProgress(int epoch, int totalEpochs,
            Map<String, Double> losses, Map<String, Double> metrics) {
        this(epoch, epoch, totalEpochs, totalEpochs, losses, metrics);
    }

    public YoloTrainingProgress(int epoch, int step, int totalEpochs,
            Map<String, Double> losses, Map<String, Double> metrics) {
        this(epoch, step, totalEpochs, totalEpochs, losses, metrics);
    }

    public YoloTrainingProgress(int epoch, int step, int totalEpochs, int totalSteps,
            Map<String, Double> losses, Map<String, Double> metrics) {
        this.epoch = epoch;
        this.step = step;
        this.totalEpochs = totalEpochs;
        this.totalSteps = totalSteps;
        this.losses = immutableCopy(losses);
        this.metrics = immutableCopy(metrics);
    }

    public int getEpoch() {
        return epoch;
    }

    public int getStep() {
        return step;
    }

    public int getTotalEpochs() {
        return totalEpochs;
    }

    public int getTotalSteps() {
        return totalSteps;
    }

    public Map<String, Double> getLosses() {
        return losses;
    }

    public Map<String, Double> getMetrics() {
        return metrics;
    }

    public Double getPrimaryLoss() {
        return firstValue(losses);
    }

    public Double getPrimaryMetric() {
        return firstValue(metrics);
    }

    public Double getTrainingTotalLoss() {
        return sumPresent(losses, TRAIN_BOX_LOSS, TRAIN_CLS_LOSS, TRAIN_DFL_LOSS);
    }

    public Double getValidationTotalLoss() {
        return sumPresent(metrics, VAL_BOX_LOSS, VAL_CLS_LOSS, VAL_DFL_LOSS);
    }

    public Double getPrimaryDetectionMetric() {
        return firstPresent(metrics, MAP50_95, MAP50, PRECISION, RECALL);
    }

    public String getPrimaryDetectionMetricName() {
        if (metrics.containsKey(MAP50_95)) {
            return PRIMARY_DETECTION_METRIC_LABEL;
        }
        if (metrics.containsKey(MAP50)) {
            return "mAP50(B)";
        }
        if (metrics.containsKey(PRECISION)) {
            return "Precision(B)";
        }
        if (metrics.containsKey(RECALL)) {
            return "Recall(B)";
        }
        return metrics.isEmpty() ? "Metric" : metrics.keySet().iterator().next();
    }

    private static Map<String, Double> immutableCopy(Map<String, Double> map) {
        if (map == null || map.isEmpty()) {
            return Collections.emptyMap();
        }
        return Collections.unmodifiableMap(new LinkedHashMap<String, Double>(map));
    }

    private static Double firstValue(Map<String, Double> map) {
        return map.isEmpty() ? null : map.values().iterator().next();
    }

    private static Double firstPresent(Map<String, Double> map, String... keys) {
        for (String key : keys) {
            Double value = map.get(key);
            if (isFinite(value)) {
                return value;
            }
        }
        return null;
    }

    private static Double sumPresent(Map<String, Double> map, String... keys) {
        double sum = 0.0d;
        int count = 0;
        for (String key : keys) {
            Double value = map.get(key);
            if (isFinite(value)) {
                sum += value.doubleValue();
                count++;
            }
        }
        return count == 0 ? null : sum;
    }

    private static boolean isFinite(Double value) {
        return value != null && !value.isNaN() && !value.isInfinite();
    }
}
