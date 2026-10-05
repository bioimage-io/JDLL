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
package io.bioimage.modelrunner.gui.custom.unet;

import java.awt.CardLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import io.bioimage.modelrunner.gui.custom.stardist.StardistValidationPreviewPanel;
import io.bioimage.modelrunner.gui.custom.yolo.YoloValidationPreviewPanel;

/** Legacy PNG previews and geometry-aware NumPy previews share the same status band. */
public class DenseValidationPreviewPanel extends YoloValidationPreviewPanel {
    private static final long serialVersionUID = 1L;
    private final YoloValidationPreviewPanel raster = new YoloValidationPreviewPanel();
    private final StardistValidationPreviewPanel volume = new StardistValidationPreviewPanel();

    public DenseValidationPreviewPanel() {
        removeAll();
        setLayout(new CardLayout());
        add(raster, "raster");
        add(volume, "volume");
    }

    @Override public void doLayout() {
        if (getLayout() != null) getLayout().layoutContainer(this);
    }

    @Override public void clearPreview() {
        if (raster == null) return;
        raster.clearPreview();
        volume.clearPreview();
    }

    @Override public void loadPreview(String path) {
        boolean arrays = false;
        try {
            JsonObject root = JsonParser.parseString(new String(Files.readAllBytes(Paths.get(path)),
                    StandardCharsets.UTF_8)).getAsJsonObject();
            arrays = root.has("items") && root.getAsJsonArray("items").size() > 0
                    && root.getAsJsonArray("items").get(0).getAsJsonObject().has("assets");
        } catch (Exception ignored) { /* The selected reader displays its error state. */ }
        ((CardLayout) getLayout()).show(this, arrays ? "volume" : "raster");
        if (arrays) volume.loadPreview(path); else raster.loadPreview(path);
    }

    @Override public void setTrainingStatus(boolean active, int step, int steps, int epochs,
            long elapsed, double secondsPerIteration) {
        if (raster == null) return;
        raster.setTrainingStatus(active, step, steps, epochs, elapsed, secondsPerIteration);
        volume.setTrainingStatus(active, step, steps, epochs, elapsed, secondsPerIteration);
    }
}
