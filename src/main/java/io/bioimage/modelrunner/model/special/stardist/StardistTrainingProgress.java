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
package io.bioimage.modelrunner.model.special.stardist;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

public final class StardistTrainingProgress {

	public static final String TRAIN_TOTAL_LOSS = "train/total_loss";
	public static final String TRAIN_PROB_LOSS = "train/prob_loss";
	public static final String TRAIN_DIST_LOSS = "train/dist_loss";
	public static final String VAL_TOTAL_LOSS = "val/total_loss";
	public static final String LEARNING_RATE = "learning_rate";
	public static final String STARDIST_TOTAL_LOSS_LABEL =
			"StarDist loss = prob_loss + dist_loss";

	private final int epoch;
	private final int step;
	private final int totalEpochs;
	private final int totalSteps;
	private final Map<String, Double> losses;
	private final Map<String, Double> metrics;

	public StardistTrainingProgress(int epoch, int step, int totalEpochs, int totalSteps,
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

	public Double getTrainingTotalLoss() {
		return firstPresent(losses, TRAIN_TOTAL_LOSS);
	}

	public Double getValidationTotalLoss() {
		return firstPresent(losses, VAL_TOTAL_LOSS);
	}

	public Double getLearningRate() {
		return firstPresent(metrics, LEARNING_RATE);
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
