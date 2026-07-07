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
package io.bioimage.modelrunner.gui.custom.training;

import java.io.File;

public final class TrainingModelPaths {

    private TrainingModelPaths() {}

    /**
     * Returns a non-existing model directory, suffixing the requested name when needed.
     *
     * @param familyDir the model family directory.
     * @param requestedName the requested model name.
     * @param directExtensions direct model-file extensions that also reserve the name.
     * @return a unique model directory.
     */
    public static File uniqueModelDir(File familyDir, String requestedName, String... directExtensions) {
        File root = familyDir == null ? new File(".") : familyDir.getAbsoluteFile();
        String baseName = requestedName == null ? "" : requestedName.trim();
        int suffix = 0;
        while (true) {
            String candidateName = suffix == 0 ? baseName : baseName + "-" + suffix;
            File candidate = new File(root, candidateName);
            if (!isOccupied(root, candidateName, candidate, directExtensions)) {
                return candidate;
            }
            suffix++;
        }
    }

    private static boolean isOccupied(File root, String candidateName, File candidate, String[] directExtensions) {
        if (candidate.exists()) {
            return true;
        }
        if (directExtensions == null) {
            return false;
        }
        for (String extension : directExtensions) {
            if (extension != null && new File(root, candidateName + extension).exists()) {
                return true;
            }
        }
        return false;
    }
}
