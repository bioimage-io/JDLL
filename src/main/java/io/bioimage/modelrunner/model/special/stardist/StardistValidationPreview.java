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

public final class StardistValidationPreview {

	private final int epoch;
	private final String previewJsonPath;

	/**
	 * Creates a new StardistValidationPreview instance.
	 *
	 * @param epoch the epoch.
	 * @param previewJsonPath the preview JSON path.
	 */
	public StardistValidationPreview(int epoch, String previewJsonPath) {
		this.epoch = epoch;
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
	 * Returns the preview JSON path.
	 *
	 * @return the preview JSON path.
	 */
	public String getPreviewJsonPath() {
		return previewJsonPath;
	}
}
