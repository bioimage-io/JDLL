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

import java.awt.Font;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;

public class YoloTrainPanel extends JPanel {

    private static final long serialVersionUID = -1127872966672515154L;

    protected static final String TRAIN_PLACEHOLDER =
            "<html><div style='text-align:center;'>Training configuration will be added here.</div></html>";

    protected final JLabel placeholderLabel = new JLabel(TRAIN_PLACEHOLDER, SwingConstants.CENTER);

    protected YoloTrainPanel() {
        setLayout(null);
        setOpaque(true);
        setBackground(YoloUiUtils.PANEL_BG);
        placeholderLabel.setFont(placeholderLabel.getFont().deriveFont(Font.PLAIN, 14f));
        add(placeholderLabel);
    }

    @Override
    public void doLayout() {
        placeholderLabel.setBounds(0, 0, getWidth(), getHeight());
        YoloUiUtils.applyResponsiveText(placeholderLabel, Math.max(1, getWidth() - 20), Math.max(24, getHeight() / 14));
    }
}
