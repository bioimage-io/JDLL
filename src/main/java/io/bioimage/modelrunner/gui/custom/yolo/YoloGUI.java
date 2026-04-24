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
import java.io.IOException;
import java.util.List;
import java.util.Map;

import javax.swing.BorderFactory;
import javax.swing.JFrame;
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
    protected final YoloTrainPanel trainPanel = new YoloTrainPanel();

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

    private static class FlatTabbedPaneUI extends BasicTabbedPaneUI {

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
                frame.getContentPane().add(new YoloGUI(adapter));
                frame.setSize(900, 900);
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
            }
        });
    }
}
