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

    /**
     * Creates a new YoloTrainingProgress instance.
     *
     * @param epoch the epoch.
     * @param totalEpochs the total epochs.
     * @param losses the losses.
     * @param metrics the metrics.
     */
    public YoloTrainingProgress(int epoch, int totalEpochs,
            Map<String, Double> losses, Map<String, Double> metrics) {
        this(epoch, epoch, totalEpochs, totalEpochs, losses, metrics);
    }

    /**
     * Creates a new YoloTrainingProgress instance.
     *
     * @param epoch the epoch.
     * @param step the step.
     * @param totalEpochs the total epochs.
     * @param losses the losses.
     * @param metrics the metrics.
     */
    public YoloTrainingProgress(int epoch, int step, int totalEpochs,
            Map<String, Double> losses, Map<String, Double> metrics) {
        this(epoch, step, totalEpochs, totalEpochs, losses, metrics);
    }

    /**
     * Creates a new YoloTrainingProgress instance.
     *
     * @param epoch the epoch.
     * @param step the step.
     * @param totalEpochs the total epochs.
     * @param totalSteps the total steps.
     * @param losses the losses.
     * @param metrics the metrics.
     */
    public YoloTrainingProgress(int epoch, int step, int totalEpochs, int totalSteps,
            Map<String, Double> losses, Map<String, Double> metrics) {
        this.epoch = epoch;
        this.step = step;
        this.totalEpochs = totalEpochs;
        this.totalSteps = totalSteps;
        this.losses = immutableCopy(losses);
        this.metrics = immutableCopy(metrics);
    }

    /**
     * Returns the epoch.
     *
     * @return the epoch.
     */
    public int getEpoch() {
        return epoch;
    }

    /**
     * Returns the step.
     *
     * @return the step.
     */
    public int getStep() {
        return step;
    }

    /**
     * Returns the total epochs.
     *
     * @return the total epochs.
     */
    public int getTotalEpochs() {
        return totalEpochs;
    }

    /**
     * Returns the total steps.
     *
     * @return the total steps.
     */
    public int getTotalSteps() {
        return totalSteps;
    }

    /**
     * Returns the losses.
     *
     * @return the losses.
     */
    public Map<String, Double> getLosses() {
        return losses;
    }

    /**
     * Returns the metrics.
     *
     * @return the metrics.
     */
    public Map<String, Double> getMetrics() {
        return metrics;
    }

    /**
     * Returns the primary loss.
     *
     * @return the primary loss.
     */
    public Double getPrimaryLoss() {
        return firstValue(losses);
    }

    /**
     * Returns the primary metric.
     *
     * @return the primary metric.
     */
    public Double getPrimaryMetric() {
        return firstValue(metrics);
    }

    /**
     * Returns the training total loss.
     *
     * @return the training total loss.
     */
    public Double getTrainingTotalLoss() {
        return sumPresent(losses, TRAIN_BOX_LOSS, TRAIN_CLS_LOSS, TRAIN_DFL_LOSS);
    }

    /**
     * Returns the validation total loss.
     *
     * @return the validation total loss.
     */
    public Double getValidationTotalLoss() {
        return sumPresent(metrics, VAL_BOX_LOSS, VAL_CLS_LOSS, VAL_DFL_LOSS);
    }

    /**
     * Returns the primary detection metric.
     *
     * @return the primary detection metric.
     */
    public Double getPrimaryDetectionMetric() {
        return firstPresent(metrics, MAP50_95, MAP50, PRECISION, RECALL);
    }

    /**
     * Returns the primary detection metric name.
     *
     * @return the primary detection metric name.
     */
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
