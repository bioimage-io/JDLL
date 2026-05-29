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

import java.awt.Color;
import java.awt.Graphics;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import javax.swing.BorderFactory;
import javax.swing.JFrame;
import javax.swing.JCheckBox;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.SwingUtilities;
import javax.swing.plaf.basic.BasicTabbedPaneUI;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.adapter.RunnerAdapter;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class StardistGUI extends JPanel {

    private static final long serialVersionUID = -7095114274286067822L;

    protected static final int TAB_PAD = 4;
    protected static final int TITLE_GAP = 6;
    protected static final double TITLE_HEIGHT_RATIO = 0.065;

    protected final JTabbedPane tabs = new JTabbedPane();
    protected final YoloTitlePanel titlePanel;
    protected final YoloInferencePanel inferencePanel = new YoloInferencePanel(false);
    protected final StardistTrainPanel trainPanel = new StardistTrainPanel();
    protected final YoloAccelerationCheckBox accelerationCheckBox = new YoloAccelerationCheckBox();

    protected StardistGUI(GuiAdapter adapter) {
        setLayout(null);
        setOpaque(true);
        setBackground(YoloUiUtils.PANEL_BG);
        this.titlePanel = new YoloTitlePanel("StarDist", adapter);
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

    @Override
    public void doLayout() {
        int titleH = Math.max(40, (int) Math.round(getHeight() * TITLE_HEIGHT_RATIO));
        titlePanel.setBounds(TAB_PAD, TAB_PAD, Math.max(0, getWidth() - TAB_PAD * 2), titleH);
        int tabsY = TAB_PAD + titleH + TITLE_GAP;
        tabs.setBounds(TAB_PAD, tabsY, Math.max(0, getWidth() - TAB_PAD * 2), Math.max(0, getHeight() - tabsY - TAB_PAD));
        YoloUiUtils.applyResponsiveFont(tabs, tabs.getBounds().height > 0 ? tabs.getBounds().height / 26 : YoloUiUtils.MIN_FONT_SIZE);
        layoutAccelerationCheckBox(tabsY);
    }

    @Override
    public boolean isOptimizedDrawingEnabled() {
        return false;
    }

    public JTabbedPane getTabs() {
        return tabs;
    }

    public YoloInferencePanel getInferencePanel() {
        return inferencePanel;
    }

    public StardistTrainPanel getTrainPanel() {
        return trainPanel;
    }

    public JCheckBox getAccelerationCheckBox() {
        return accelerationCheckBox;
    }

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

        @Override
        protected void paintTabBackground(Graphics g, int tabPlacement, int tabIndex,
                int x, int y, int w, int h, boolean isSelected) {
            g.setColor(isSelected ? YoloUiUtils.PANEL_BG : YoloUiUtils.PRIMARY_BUTTON_BG);
            g.fillRect(x, y, w, h);
        }

        @Override
        protected void paintContentBorder(java.awt.Graphics g, int tabPlacement, int selectedIndex) {
        }

        @Override
        protected void paintFocusIndicator(java.awt.Graphics g, int tabPlacement, java.awt.Rectangle[] rects,
                int tabIndex, java.awt.Rectangle iconRect, java.awt.Rectangle textRect, boolean isSelected) {
        }
    }

    public static void main(String[] args) {
    	
    	GuiAdapter adapter = new GuiAdapter() {

			@Override
			public String getSoftwareName() {
				return "JDLL";
			}

			@Override
			public String getSoftwareDescription() {
				return "";
			}

			@Override
			public Color getTitleColor() {
				return new Color(200, 100, 100);
			}

			@Override
			public Color getSubtitleColor() {
				return null;
			}

			@Override
			public Color getHeaderColor() {
				return null;
			}

			@Override
			public String getIconPath() {
				return null;
			}

			@Override
			public String getModelsDir() {
				return null;
			}

			@Override
			public String getEnginesDir() {
				return null;
			}

			@Override
			public RunnerAdapter createRunner(ModelDescriptor descriptor) throws IOException, LoadEngineException {
				// TODO Auto-generated method stub
				return null;
			}

			@Override
			public RunnerAdapter createRunner(ModelDescriptor descriptor, String enginesPath)
					throws IOException, LoadEngineException {
				// TODO Auto-generated method stub
				return null;
			}

			@Override
			public <T extends RealType<T> & NativeType<T>> void displayRai(RandomAccessibleInterval<T> rai,
					String axesOrder, String imTitle) {
				// TODO Auto-generated method stub
				
			}

			@Override
			public <T extends RealType<T> & NativeType<T>> List<Tensor<T>> getInputTensors(ModelDescriptor descriptor) {
				// TODO Auto-generated method stub
				return null;
			}

			@Override
			public List<String> getInputImageNames() {
				// TODO Auto-generated method stub
				return null;
			}

			@Override
			public <T extends RealType<T> & NativeType<T>> List<Tensor<T>> convertToInputTensors(
					Map<String, Object> inputs, ModelDescriptor descriptor) {
				// TODO Auto-generated method stub
				return null;
			}

			@Override
			public void notifyModelUsed(String modelAbsPath) {
				// TODO Auto-generated method stub
				
			}
    		
    	};
    	
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                JFrame frame = new JFrame("YOLO Plugin");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(new StardistGUI(adapter));
                frame.setSize(900, 900);
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
            }
        });
    }
}
