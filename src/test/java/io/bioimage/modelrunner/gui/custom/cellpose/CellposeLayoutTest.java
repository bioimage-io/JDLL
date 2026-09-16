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
package io.bioimage.modelrunner.gui.custom.cellpose;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.awt.Component;
import java.awt.Container;

import org.junit.Test;

import io.bioimage.modelrunner.gui.custom.CellposePluginUI;

public class CellposeLayoutTest {

    @Test
    public void usesSingleStandardInferenceTab() {
        CellposePluginUI gui = new CellposePluginUI(null, null);

        assertEquals(1, gui.getTabs().getTabCount());
        assertEquals("Inference", gui.getTabs().getTitleAt(0));
        assertNull(gui.getTrainPanel());
    }

    @Test
    public void placesCellposeOptionsBetweenImageSelectionAndPreview() {
        CellposePluginUI gui = new CellposePluginUI(null, null);
        gui.setSize(550, 700);
        layoutTree(gui);

        CellposeInferencePanel inference = gui.getInferencePanel();
        assertTrue(inference.getOptionsPanel().getY()
                > inference.getImageSourcePanel().getY());
        assertTrue(inference.getImageDisplayPanel().getY()
                > inference.getOptionsPanel().getY());
        assertTrue(inference.getOptionsPanel().getCytoplasmComboBox().getWidth() > 0);
        assertTrue(inference.getOptionsPanel().getNucleiComboBox().getWidth() > 0);
        assertTrue(inference.getOptionsPanel().getDisplayIntermediateOutputsCheckBox().getWidth() > 0);
        assertTrue(inference.getActionPanel().getY() + inference.getActionPanel().getHeight()
                <= inference.getHeight());
    }

    private static void layoutTree(Container container) {
        container.doLayout();
        for (Component child : container.getComponents()) {
            if (child instanceof Container) {
                layoutTree((Container) child);
            }
        }
    }
}
