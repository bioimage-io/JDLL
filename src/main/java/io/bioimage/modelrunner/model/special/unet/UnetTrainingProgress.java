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
package io.bioimage.modelrunner.model.special.unet;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

public final class UnetTrainingProgress {

    public static final String TRAIN_TOTAL_LOSS = "train/total_loss";
    public static final String VAL_TOTAL_LOSS = "val/total_loss";
    public static final String VAL_DICE = "val/dice";
    public static final String VAL_MEAN_DICE = "val/mean_dice";
    public static final String VAL_FOREGROUND_DICE = "val/foreground_dice";
    public static final String VAL_IOU = "val/iou";
    public static final String UNET_TOTAL_LOSS_LABEL = "UNet loss";

    private final int epoch;
    private final int step;
    private final int totalEpochs;
    private final int totalSteps;
    private final Map<String, Double> losses;
    private final Map<String, Double> metrics;

    /**
     * Creates a new UnetTrainingProgress instance.
     *
     * @param epoch the epoch.
     * @param step the step.
     * @param totalEpochs the total epochs.
     * @param totalSteps the total steps.
     * @param losses the losses.
     * @param metrics the metrics.
     */
    public UnetTrainingProgress(int epoch, int step, int totalEpochs, int totalSteps,
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
     * Returns the training total loss.
     *
     * @return the training total loss.
     */
    public Double getTrainingTotalLoss() {
        return firstPresent(losses, TRAIN_TOTAL_LOSS);
    }

    /**
     * Returns the validation total loss.
     *
     * @return the validation total loss.
     */
    public Double getValidationTotalLoss() {
        return firstPresent(losses, VAL_TOTAL_LOSS);
    }

    /**
     * Returns the primary validation metric.
     *
     * @return the primary validation metric.
     */
    public Double getPrimaryMetric() {
        Double dice = firstPresent(metrics, VAL_DICE);
        if (dice != null) {
            return dice;
        }
        dice = firstPresent(metrics, VAL_MEAN_DICE);
        if (dice != null) {
            return dice;
        }
        dice = firstPresent(metrics, VAL_FOREGROUND_DICE);
        return dice != null ? dice : firstPresent(metrics, VAL_IOU);
    }

    /**
     * Returns the primary validation metric name.
     *
     * @return the metric name.
     */
    public String getPrimaryMetricName() {
        if (metrics.containsKey(VAL_DICE)) {
            return "Dice";
        }
        if (metrics.containsKey(VAL_MEAN_DICE)) {
            return "Mean Dice";
        }
        if (metrics.containsKey(VAL_FOREGROUND_DICE)) {
            return "Foreground Dice";
        }
        if (metrics.containsKey(VAL_IOU)) {
            return "IoU";
        }
        return metrics.isEmpty() ? "Metric" : metrics.keySet().iterator().next();
    }

    private static Map<String, Double> immutableCopy(Map<String, Double> map) {
        if (map == null || map.isEmpty()) {
            return Collections.emptyMap();
        }
        return Collections.unmodifiableMap(new LinkedHashMap<String, Double>(map));
    }

    private static Double firstPresent(Map<String, Double> map, String key) {
        Double value = map.get(key);
        return value != null && !value.isNaN() && !value.isInfinite() ? value : null;
    }
}
