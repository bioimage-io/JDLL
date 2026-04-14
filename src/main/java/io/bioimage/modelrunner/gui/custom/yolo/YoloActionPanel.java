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
package io.bioimage.modelrunner.gui.custom.yolo;

import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;

import javax.swing.JButton;
import javax.swing.JPanel;

public class YoloActionPanel extends JPanel {

    private static final long serialVersionUID = -763353214342801966L;

    private static final int GAP = 6;

    protected final JButton cancelButton = new JButton("Cancel");
    protected final JButton runButton = new JButton("Infer");

    protected YoloActionPanel() {
        setLayout(null);
        setOpaque(false);
        YoloUiUtils.styleFlatSecondaryButton(cancelButton);
        YoloUiUtils.styleFlatSecondaryButton(runButton);
        add(cancelButton);
        add(runButton);
        addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                doLayout();
            }
        });
    }

    @Override
    public void doLayout() {
        int w = Math.max(0, getWidth());
        int h = Math.max(0, getHeight());
        int cancelW = Math.max(1, (w - GAP) / 2);
        int runW = Math.max(1, w - GAP - cancelW);
        cancelButton.setBounds(0, 0, cancelW, h);
        runButton.setBounds(cancelW + GAP, 0, runW, h);
        YoloUiUtils.applyResponsiveText(cancelButton, cancelW - 8, h);
        YoloUiUtils.applyResponsiveText(runButton, runW - 8, h);
    }

    public JButton getCancelButton() {
        return cancelButton;
    }

    public JButton getRunButton() {
        return runButton;
    }
}
