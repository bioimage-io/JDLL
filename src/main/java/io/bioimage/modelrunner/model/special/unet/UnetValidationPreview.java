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

/**
 * UNet validation preview metadata emitted during training.
 */
public final class UnetValidationPreview {

    private final int epoch;
    private final String previewJsonPath;
    private final String latestPreviewJsonPath;

    /**
     * Creates a new UnetValidationPreview instance.
     *
     * @param epoch the epoch.
     * @param previewJsonPath the epoch-specific preview JSON path.
     * @param latestPreviewJsonPath the latest preview JSON path.
     */
    public UnetValidationPreview(int epoch, String previewJsonPath, String latestPreviewJsonPath) {
        this.epoch = epoch;
        this.previewJsonPath = previewJsonPath;
        this.latestPreviewJsonPath = latestPreviewJsonPath;
    }

    /**
     * Gets the epoch.
     *
     * @return the epoch.
     */
    public int getEpoch() {
        return epoch;
    }

    /**
     * Gets the preview JSON path.
     *
     * @return the preview JSON path.
     */
    public String getPreviewJsonPath() {
        return previewJsonPath;
    }

    /**
     * Gets the latest preview JSON path.
     *
     * @return the latest preview JSON path.
     */
    public String getLatestPreviewJsonPath() {
        return latestPreviewJsonPath;
    }
}
