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

public final class YoloValidationPreview {

    private final int epoch;
    private final String checkpointPath;
    private final String previewJsonPath;

    /**
     * Creates a new YoloValidationPreview instance.
     *
     * @param epoch the epoch.
     * @param checkpointPath the checkpoint path.
     */
    public YoloValidationPreview(int epoch, String checkpointPath) {
        this(epoch, checkpointPath, null);
    }

    /**
     * Creates a new YoloValidationPreview instance.
     *
     * @param epoch the epoch.
     * @param checkpointPath the checkpoint path.
     * @param previewJsonPath the preview JSON path.
     */
    public YoloValidationPreview(int epoch, String checkpointPath, String previewJsonPath) {
        this.epoch = epoch;
        this.checkpointPath = checkpointPath;
        this.previewJsonPath = previewJsonPath;
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
     * Returns the checkpoint path.
     *
     * @return the checkpoint path.
     */
    public String getCheckpointPath() {
        return checkpointPath;
    }

    /**
     * Returns the preview JSON path.
     *
     * @return the preview JSON path.
     */
    public String getPreviewJsonPath() {
        return previewJsonPath;
    }
}
