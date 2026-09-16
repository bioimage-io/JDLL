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
package io.bioimage.modelrunner.gui.custom.gui;

import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.custom.cellpose.CellposeInferencePanel;
import io.bioimage.modelrunner.gui.custom.yolo.YoloGUI;

/**
 * Standard inference-only Cellpose GUI.
 */
public class CellposeGUI extends YoloGUI {

    private static final long serialVersionUID = 5381352117710530216L;

    protected final CellposeInferencePanel cellposeInferencePanel;

    /**
     * Creates a standalone Cellpose GUI.
     */
    protected CellposeGUI() {
        this(null);
    }

    /**
     * Creates a Cellpose GUI for a host application.
     *
     * @param adapter the host GUI adapter.
     */
    protected CellposeGUI(GuiAdapter adapter) {
        this(adapter, new CellposeInferencePanel());
    }

    private CellposeGUI(GuiAdapter adapter, CellposeInferencePanel inferencePanel) {
        super(adapter, "Cellpose", inferencePanel, null);
        this.cellposeInferencePanel = inferencePanel;
    }

    /**
     * Returns the Cellpose inference panel.
     *
     * @return the inference panel.
     */
    @Override
    public CellposeInferencePanel getInferencePanel() {
        return cellposeInferencePanel;
    }
}
