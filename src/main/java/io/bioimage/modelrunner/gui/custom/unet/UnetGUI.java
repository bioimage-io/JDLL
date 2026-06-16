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

import java.awt.Graphics;

import javax.swing.BorderFactory;
import javax.swing.JCheckBox;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.plaf.basic.BasicTabbedPaneUI;

import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.custom.yolo.YoloAccelerationCheckBox;
import io.bioimage.modelrunner.gui.custom.yolo.YoloTitlePanel;
import io.bioimage.modelrunner.gui.custom.yolo.YoloUiUtils;

public class UnetGUI extends JPanel {

    private static final long serialVersionUID = 6867124250927343385L;

    protected static final int TAB_PAD = 4;
    protected static final int TITLE_GAP = 6;
    protected static final double TITLE_HEIGHT_RATIO = 0.065;

    protected final JTabbedPane tabs = new JTabbedPane();
    protected final YoloTitlePanel titlePanel;
    protected final UnetInferencePanel inferencePanel = new UnetInferencePanel();
    protected final UnetTrainPanel trainPanel = new UnetTrainPanel();
    protected final YoloAccelerationCheckBox accelerationCheckBox = new YoloAccelerationCheckBox();

    /**
     * Creates a new UnetGUI instance.
     *
     * @param adapter the adapter.
     */
    protected UnetGUI(GuiAdapter adapter) {
        setLayout(null);
        setOpaque(true);
        setBackground(YoloUiUtils.PANEL_BG);
        this.titlePanel = new YoloTitlePanel("UNet", adapter);
        tabs.setBorder(BorderFactory.createEmptyBorder());
        tabs.setOpaque(true);
        tabs.setBackground(YoloUiUtils.PANEL_BG);
        tabs.setUI(new FlatTabbedPaneUI());
        tabs.addTab("Inference", inferencePanel);
        tabs.addTab("Train", trainPanel);
        tabs.addChangeListener(e -> tabs.repaint());
        add(titlePanel);
        add(tabs);
        add(accelerationCheckBox);
        setComponentZOrder(accelerationCheckBox, 0);
    }

    /**
     * Performs do layout.
     */
    @Override
    public void doLayout() {
        int titleH = Math.max(40, (int) Math.round(getHeight() * TITLE_HEIGHT_RATIO));
        titlePanel.setBounds(TAB_PAD, TAB_PAD, Math.max(0, getWidth() - TAB_PAD * 2), titleH);
        int tabsY = TAB_PAD + titleH + TITLE_GAP;
        tabs.setBounds(TAB_PAD, tabsY, Math.max(0, getWidth() - TAB_PAD * 2), Math.max(0, getHeight() - tabsY - TAB_PAD));
        YoloUiUtils.applyResponsiveFont(tabs,
                tabs.getBounds().height > 0 ? tabs.getBounds().height / 26 : YoloUiUtils.MIN_FONT_SIZE);
        layoutAccelerationCheckBox(tabsY);
    }

    /**
     * Returns whether optimized drawing enabled.
     *
     * @return true if optimized drawing enabled; false otherwise.
     */
    @Override
    public boolean isOptimizedDrawingEnabled() {
        return false;
    }

    /**
     * Returns the tabs.
     *
     * @return the tabs.
     */
    public JTabbedPane getTabs() {
        return tabs;
    }

    /**
     * Returns the inference panel.
     *
     * @return the inference panel.
     */
    public UnetInferencePanel getInferencePanel() {
        return inferencePanel;
    }

    /**
     * Returns the train panel.
     *
     * @return the train panel.
     */
    public UnetTrainPanel getTrainPanel() {
        return trainPanel;
    }

    /**
     * Returns the acceleration check box.
     *
     * @return the acceleration check box.
     */
    public JCheckBox getAccelerationCheckBox() {
        return accelerationCheckBox;
    }

    /**
     * Returns whether acceleration enabled.
     *
     * @return true if acceleration enabled; false otherwise.
     */
    public boolean isAccelerationEnabled() {
        return accelerationCheckBox.isSelected();
    }

    private void layoutAccelerationCheckBox(int tabsY) {
        int tabHeaderH = Math.max(YoloUiUtils.controlHeightForFontSize(tabs.getFont().getSize()) + 4,
                tabs.getFont().getSize() + 12);
        int h = Math.max(18, tabHeaderH - 6);
        int preferredW = accelerationCheckBox.getPreferredSize().width + 12;
        int maxW = Math.max(0, tabs.getWidth() / 2);
        int w = Math.min(Math.max(116, preferredW), maxW);
        int x = TAB_PAD + tabs.getWidth() - w - 8;
        int y = tabsY + Math.max(2, (tabHeaderH - h) / 2);
        accelerationCheckBox.setBounds(Math.max(TAB_PAD, x), y, w, h);
        YoloUiUtils.applyResponsiveText(accelerationCheckBox, w, h);
        accelerationCheckBox.repaint();
    }

    private static class FlatTabbedPaneUI extends BasicTabbedPaneUI {

        /**
         * Performs paint tab background.
         *
         * @param g the g.
         * @param tabPlacement the tab placement.
         * @param tabIndex the tab index.
         * @param x the x.
         * @param y the y.
         * @param w the w.
         * @param h the h.
         * @param isSelected whether to use is selected.
         */
        @Override
        protected void paintTabBackground(Graphics g, int tabPlacement, int tabIndex,
                int x, int y, int w, int h, boolean isSelected) {
            g.setColor(isSelected ? YoloUiUtils.PANEL_BG : YoloUiUtils.PRIMARY_BUTTON_BG);
            g.fillRect(x, y, w, h);
        }

        /**
         * Performs paint content border.
         *
         * @param g the g.
         * @param tabPlacement the tab placement.
         * @param selectedIndex the selected index.
         */
        @Override
        protected void paintContentBorder(Graphics g, int tabPlacement, int selectedIndex) {
        }

        /**
         * Performs paint focus indicator.
         *
         * @param g the g.
         * @param tabPlacement the tab placement.
         * @param rects the rects.
         * @param tabIndex the tab index.
         * @param iconRect the icon rect.
         * @param textRect the text rect.
         * @param isSelected whether to use is selected.
         */
        @Override
        protected void paintFocusIndicator(Graphics g, int tabPlacement, java.awt.Rectangle[] rects,
                int tabIndex, java.awt.Rectangle iconRect, java.awt.Rectangle textRect, boolean isSelected) {
        }
    }
}
