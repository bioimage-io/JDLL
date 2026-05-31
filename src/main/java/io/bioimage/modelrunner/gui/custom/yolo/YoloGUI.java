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

public class YoloGUI extends JPanel {

    private static final long serialVersionUID = -7095114274286067822L;

    protected static final int TAB_PAD = 4;
    protected static final int TITLE_GAP = 6;
    protected static final double TITLE_HEIGHT_RATIO = 0.065;

    protected final JTabbedPane tabs = new JTabbedPane();
    protected final YoloTitlePanel titlePanel;
    protected final YoloInferencePanel inferencePanel = new YoloInferencePanel();
    protected final BaseTrainPanel trainPanel = new YoloTrainPanel();
    protected final YoloAccelerationCheckBox accelerationCheckBox = new YoloAccelerationCheckBox();

    /**
     * Creates a new YoloGUI instance.
     *
     * @param adapter the adapter.
     */
    protected YoloGUI(GuiAdapter adapter) {
        setLayout(null);
        setOpaque(true);
        setBackground(YoloUiUtils.PANEL_BG);
        this.titlePanel = new YoloTitlePanel(adapter);
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
        YoloUiUtils.applyResponsiveFont(tabs, tabs.getBounds().height > 0 ? tabs.getBounds().height / 26 : YoloUiUtils.MIN_FONT_SIZE);
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
    public YoloInferencePanel getInferencePanel() {
        return inferencePanel;
    }

    /**
     * Returns the train panel.
     *
     * @return the train panel.
     */
    public BaseTrainPanel getTrainPanel() {
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
        protected void paintContentBorder(java.awt.Graphics g, int tabPlacement, int selectedIndex) {
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
        protected void paintFocusIndicator(java.awt.Graphics g, int tabPlacement, java.awt.Rectangle[] rects,
                int tabIndex, java.awt.Rectangle iconRect, java.awt.Rectangle textRect, boolean isSelected) {
        }
    }

    /**
     * Runs this class from the command line.
     *
     * @param args command-line arguments.
     */
    public static void main(String[] args) {
    	
    	GuiAdapter adapter = new GuiAdapter() {

			/**
			 * Returns the software name.
			 *
			 * @return the software name.
			 */
			@Override
			public String getSoftwareName() {
				return "JDLL";
			}

			/**
			 * Returns the software description.
			 *
			 * @return the software description.
			 */
			@Override
			public String getSoftwareDescription() {
				return "";
			}

			/**
			 * Returns the title color.
			 *
			 * @return the title color.
			 */
			@Override
			public Color getTitleColor() {
				return new Color(200, 100, 100);
			}

			/**
			 * Returns the subtitle color.
			 *
			 * @return the subtitle color.
			 */
			@Override
			public Color getSubtitleColor() {
				return null;
			}

			/**
			 * Returns the header color.
			 *
			 * @return the header color.
			 */
			@Override
			public Color getHeaderColor() {
				return null;
			}

			/**
			 * Returns the icon path.
			 *
			 * @return the icon path.
			 */
			@Override
			public String getIconPath() {
				return null;
			}

			/**
			 * Returns the models directory.
			 *
			 * @return the models directory.
			 */
			@Override
			public String getModelsDir() {
				return null;
			}

			/**
			 * Returns the engines directory.
			 *
			 * @return the engines directory.
			 */
			@Override
			public String getEnginesDir() {
				return null;
			}

			/**
			 * Creates the runner.
			 *
			 * @param descriptor the descriptor.
			 * @return the created runner adapter.
			 * @throws IOException if an I/O error occurs.
			 * @throws LoadEngineException if the engine cannot be loaded.
			 */
			@Override
			public RunnerAdapter createRunner(ModelDescriptor descriptor) throws IOException, LoadEngineException {
				// TODO Auto-generated method stub
				return null;
			}

			/**
			 * Creates the runner.
			 *
			 * @param descriptor the descriptor.
			 * @param enginesPath the engines path.
			 * @return the created runner adapter.
			 * @throws IOException if an I/O error occurs.
			 * @throws LoadEngineException if the engine cannot be loaded.
			 */
			@Override
			public RunnerAdapter createRunner(ModelDescriptor descriptor, String enginesPath)
					throws IOException, LoadEngineException {
				// TODO Auto-generated method stub
				return null;
			}

			/**
			 * Performs display RAI.
			 *
			 * @param <T> the T type parameter.
			 * @param rai the RAI.
			 * @param axesOrder the axes order.
			 * @param imTitle the image title.
			 */
			@Override
			public <T extends RealType<T> & NativeType<T>> void displayRai(RandomAccessibleInterval<T> rai,
					String axesOrder, String imTitle) {
				// TODO Auto-generated method stub
				
			}

			/**
			 * Returns the input tensors.
			 *
			 * @param <T> the T type parameter.
			 * @param descriptor the descriptor.
			 * @return the input tensors.
			 */
			@Override
			public <T extends RealType<T> & NativeType<T>> List<Tensor<T>> getInputTensors(ModelDescriptor descriptor) {
				// TODO Auto-generated method stub
				return null;
			}

			/**
			 * Returns the input image names.
			 *
			 * @return the input image names.
			 */
			@Override
			public List<String> getInputImageNames() {
				// TODO Auto-generated method stub
				return null;
			}

			/**
			 * Returns the result of convert to input tensors.
			 *
			 * @param <T> the T type parameter.
			 * @param inputs the inputs to process.
			 * @param descriptor the descriptor.
			 * @return the resulting list.
			 */
			@Override
			public <T extends RealType<T> & NativeType<T>> List<Tensor<T>> convertToInputTensors(
					Map<String, Object> inputs, ModelDescriptor descriptor) {
				// TODO Auto-generated method stub
				return null;
			}

			/**
			 * Performs notify model used.
			 *
			 * @param modelAbsPath the model abs path.
			 */
			@Override
			public void notifyModelUsed(String modelAbsPath) {
				// TODO Auto-generated method stub
				
			}
    		
    	};
    	
        SwingUtilities.invokeLater(new Runnable() {
            /**
             * Runs the run.
             */
            @Override
            public void run() {
                JFrame frame = new JFrame("YOLO Plugin");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(new YoloGUI(adapter));
                frame.setSize(900, 900);
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
            }
        });
    }
}
