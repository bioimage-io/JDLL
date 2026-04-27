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

    private final int epoch;
    private final int totalEpochs;
    private final Map<String, Double> losses;
    private final Map<String, Double> metrics;

    public YoloTrainingProgress(int epoch, int totalEpochs,
            Map<String, Double> losses, Map<String, Double> metrics) {
        this.epoch = epoch;
        this.totalEpochs = totalEpochs;
        this.losses = immutableCopy(losses);
        this.metrics = immutableCopy(metrics);
    }

    public int getEpoch() {
        return epoch;
    }

    public int getTotalEpochs() {
        return totalEpochs;
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

    private static Map<String, Double> immutableCopy(Map<String, Double> map) {
        if (map == null || map.isEmpty()) {
            return Collections.emptyMap();
        }
        return Collections.unmodifiableMap(new LinkedHashMap<String, Double>(map));
    }

    private static Double firstValue(Map<String, Double> map) {
        return map.isEmpty() ? null : map.values().iterator().next();
    }
}
