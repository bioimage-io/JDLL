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
package io.bioimage.modelrunner.gui.yolo;

import java.util.Arrays;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.SwingUtilities;

public class YoloGUI extends JPanel {

    private static final long serialVersionUID = -7095114274286067822L;

    protected static final int TAB_PAD = 4;
    protected static final int TITLE_GAP = 6;
    protected static final double TITLE_HEIGHT_RATIO = 0.065;

    protected static final List<String> DEFAULT_MODELS = Arrays.asList(
            "[Pretrained] YOLO11n",
            "[Pretrained] YOLO11s",
            "[Pretrained] YOLO11m",
            "[Pretrained] YOLO11l",
            "[Pretrained] YOLO11x");

    protected final JTabbedPane tabs = new JTabbedPane();
    protected final YoloTitlePanel titlePanel = new YoloTitlePanel();
    protected final YoloInferencePanel inferencePanel = new YoloInferencePanel();
    protected final YoloTrainPanel trainPanel = new YoloTrainPanel();

    protected YoloGUI() {
        setLayout(null);
        setOpaque(true);
        setBackground(YoloUiUtils.PANEL_BG);
        inferencePanel.getModelSelectionPanel().setModels(DEFAULT_MODELS);
        tabs.addTab("Inference", inferencePanel);
        tabs.addTab("Train", trainPanel);
        add(titlePanel);
        add(tabs);
    }

    @Override
    public void doLayout() {
        int titleH = Math.max(40, (int) Math.round(getHeight() * TITLE_HEIGHT_RATIO));
        titlePanel.setBounds(TAB_PAD, TAB_PAD, Math.max(0, getWidth() - TAB_PAD * 2), titleH);
        int tabsY = TAB_PAD + titleH + TITLE_GAP;
        tabs.setBounds(TAB_PAD, tabsY, Math.max(0, getWidth() - TAB_PAD * 2), Math.max(0, getHeight() - tabsY - TAB_PAD));
        YoloUiUtils.applyResponsiveFont(tabs, tabs.getBounds().height > 0 ? tabs.getBounds().height / 26 : YoloUiUtils.MIN_FONT_SIZE);
    }

    public JTabbedPane getTabs() {
        return tabs;
    }

    public YoloInferencePanel getInferencePanel() {
        return inferencePanel;
    }

    public YoloTrainPanel getTrainPanel() {
        return trainPanel;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                JFrame frame = new JFrame("YOLO Plugin");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(new YoloGUI());
                frame.setSize(900, 900);
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
            }
        });
    }
}
