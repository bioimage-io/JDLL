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
package io.bioimage.modelrunner.model;

/**
 * Structured progress event emitted by model inference code.
 * <p>
 * The model layer owns the timing and numeric state. Service/UI layers should
 * translate these events into user-facing messages.
 */
public final class InferenceProgress {

    public enum Phase {
        MODEL_LOADING,
        MODEL_LOADED,
        INFERENCE_START,
        PATCH_START,
        PATCH_END,
        TASK_RETRY,
        MERGE_START,
        INFERENCE_END
    }

    private final Phase phase;
    private final int patchIndex;
    private final int totalPatches;
    private final String detail;

    private InferenceProgress(final Phase phase, final int patchIndex, final int totalPatches,
            final String detail) {
        this.phase = phase;
        this.patchIndex = patchIndex;
        this.totalPatches = totalPatches;
        this.detail = detail;
    }

    public static InferenceProgress modelLoading(final String detail) {
        return new InferenceProgress(Phase.MODEL_LOADING, -1, -1, detail);
    }

    public static InferenceProgress modelLoaded(final String detail) {
        return new InferenceProgress(Phase.MODEL_LOADED, -1, -1, detail);
    }

    public static InferenceProgress inferenceStart(final int totalPatches) {
        return new InferenceProgress(Phase.INFERENCE_START, 0, totalPatches, null);
    }

    public static InferenceProgress patchStart(final int patchIndex, final int totalPatches) {
        return new InferenceProgress(Phase.PATCH_START, patchIndex, totalPatches, null);
    }

    public static InferenceProgress patchEnd(final int patchIndex, final int totalPatches) {
        return new InferenceProgress(Phase.PATCH_END, patchIndex, totalPatches, null);
    }

    public static InferenceProgress taskRetry(final String detail) {
        return new InferenceProgress(Phase.TASK_RETRY, -1, -1, detail);
    }

    public static InferenceProgress mergeStart() {
        return new InferenceProgress(Phase.MERGE_START, -1, -1, null);
    }

    public static InferenceProgress inferenceEnd() {
        return new InferenceProgress(Phase.INFERENCE_END, -1, -1, null);
    }

    public Phase getPhase() {
        return phase;
    }

    public int getPatchIndex() {
        return patchIndex;
    }

    public int getTotalPatches() {
        return totalPatches;
    }

    public String getDetail() {
        return detail;
    }
}
