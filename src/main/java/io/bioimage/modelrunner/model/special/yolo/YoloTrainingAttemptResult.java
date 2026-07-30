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

/**
 * Runtime information produced by one isolated YOLO training attempt.
 */
public final class YoloTrainingAttemptResult {

    private String pythonVersion;
    private String ultralyticsVersion;
    private String torchVersion;
    private String torchvisionVersion;
    private String resolvedDevice;
    private Integer resolvedBatch;
    private int completedEpochs;
    private int bestEpoch;
    private Map<String, Double> bestMetrics = Collections.emptyMap();
    private Map<String, Double> finalMetrics = Collections.emptyMap();

    void updateRuntime(Map<String, Object> info) {
        pythonVersion = stringValue(info.get("python_version"), pythonVersion);
        ultralyticsVersion = stringValue(info.get("ultralytics_version"), ultralyticsVersion);
        torchVersion = stringValue(info.get("torch_version"), torchVersion);
        torchvisionVersion = stringValue(info.get("torchvision_version"), torchvisionVersion);
        resolvedDevice = stringValue(info.get("resolved_device"), resolvedDevice);
        resolvedBatch = integerValue(info.get("resolved_batch"), resolvedBatch);
    }

    void updateProgress(Map<String, Object> info) {
        updateRuntime(info);
        if (!Boolean.TRUE.equals(info.get("epoch_complete"))) {
            return;
        }
        int epoch = integerValue(info.get("epoch"), Integer.valueOf(0)).intValue();
        completedEpochs = Math.max(completedEpochs, epoch);
        finalMetrics = doubleMap(info.get("metrics"));
        Double score = finalMetrics.get("metrics/mAP50-95(B)");
        Double bestScore = bestMetrics.get("metrics/mAP50-95(B)");
        if (score != null && (bestScore == null || score.doubleValue() > bestScore.doubleValue())) {
            bestEpoch = epoch;
            bestMetrics = finalMetrics;
        }
    }

    void complete(Map<String, Object> outputs) {
        updateRuntime(outputs);
        completedEpochs = integerValue(outputs.get("completed_epochs"), Integer.valueOf(completedEpochs));
        bestEpoch = integerValue(outputs.get("best_epoch"), Integer.valueOf(bestEpoch));
        bestMetrics = doubleMap(outputs.get("best_metrics"));
        finalMetrics = doubleMap(outputs.get("final_metrics"));
    }

    public String getPythonVersion() {
        return pythonVersion;
    }

    public String getUltralyticsVersion() {
        return ultralyticsVersion;
    }

    public String getTorchVersion() {
        return torchVersion;
    }

    public String getTorchvisionVersion() {
        return torchvisionVersion;
    }

    public String getResolvedDevice() {
        return resolvedDevice;
    }

    public Integer getResolvedBatch() {
        return resolvedBatch;
    }

    public int getCompletedEpochs() {
        return completedEpochs;
    }

    public int getBestEpoch() {
        return bestEpoch;
    }

    public Map<String, Double> getBestMetrics() {
        return bestMetrics;
    }

    public Map<String, Double> getFinalMetrics() {
        return finalMetrics;
    }

    private static String stringValue(Object value, String fallback) {
        return value == null ? fallback : value.toString();
    }

    private static Integer integerValue(Object value, Integer fallback) {
        if (value instanceof Number) {
            return Integer.valueOf(((Number) value).intValue());
        }
        try {
            return value == null ? fallback : Integer.valueOf(value.toString());
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static Map<String, Double> doubleMap(Object value) {
        if (!(value instanceof Map)) {
            return Collections.emptyMap();
        }
        Map<String, Double> result = new LinkedHashMap<String, Double>();
        for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
            if (entry.getValue() instanceof Number) {
                result.put(String.valueOf(entry.getKey()),
                        Double.valueOf(((Number) entry.getValue()).doubleValue()));
            }
        }
        return Collections.unmodifiableMap(result);
    }
}
