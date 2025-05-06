/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.gui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.Insets;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.border.Border;
import javax.swing.border.TitledBorder;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.adapter.RunnerAdapter;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class ModelSelectionPanelGui extends JPanel {

	private static final long serialVersionUID = 6264134076603842497L;

	protected ModelCard prevModelPanel;
    protected ModelCard selectedModelPanel;
    protected ModelCard nextModelPanel;
    
    protected JButton nextButton;
    protected JButton prevButton;
    protected TitledBorder lineBorder;
    
    protected String defaultString = Gui.LOADING_STR;

    protected static final double MAIN_CARD_RT = 1;
    protected static final double SECOND_CARD_RT = 0.6;

    protected static String MAIN_CARD_ID = "main";
    protected static String PREV_CARD_ID = "prev";
    protected static String NEXT_CARD_ID = "next";

    protected static final double BTN_HEIGHT_RATIO = 0.07;
    protected static final double MAX_BTN_HEIGHT = 33;


	protected ModelSelectionPanelGui(GuiAdapter adapter) {
        super(null);
        this.setBackground(new Color(236, 240, 241));
        lineBorder = BorderFactory.createTitledBorder(BorderFactory.createLineBorder(Color.gray, 2, true), 
        		Gui.LOCAL_STR);
        Border paddingBorder = BorderFactory.createEmptyBorder(2, 2, 2, 2);
        this.setBorder(BorderFactory.createCompoundBorder(paddingBorder,lineBorder));

        prevModelPanel = ModelCard.createModelCard(adapter, PREV_CARD_ID, SECOND_CARD_RT);
        selectedModelPanel = ModelCard.createModelCard(adapter, MAIN_CARD_ID, MAIN_CARD_RT);
        nextModelPanel = ModelCard.createModelCard(adapter, NEXT_CARD_ID, SECOND_CARD_RT);


        prevButton = new JButton("◀");
        prevButton.setFont(new Font("SansSerif", Font.BOLD, 10));

        nextButton = new JButton("▶");
        nextButton.setFont(new Font("SansSerif", Font.BOLD, 10));
        
        add(prevModelPanel);
        add(selectedModelPanel);
        add(nextModelPanel);
        add(prevButton);
        add(nextButton);
        
        organiseComponents();
	}
    
    private void organiseComponents() {
    	addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                Insets in = getInsets();
                int W = getWidth()  - in.left - in.right;
                int rawW = getWidth();
                int rawH = getHeight();
                int inset = in.left;
                int insetTop = in.top;
                
                double btnH = Math.min(MAX_BTN_HEIGHT, rawH * BTN_HEIGHT_RATIO);
                btnH = Math.max(1, btnH);
                int btnW = (int) Math.max(1,  W * 0.5);
                prevButton.setBounds(in.left, rawH - in.left - (int) btnH, btnW, (int) btnH);
                nextButton.setBounds(in.left + btnW, rawH - in.left - (int) btnH, btnW, (int) btnH);
                
                int hSide = (int) (rawH - inset - insetTop - btnH);
                int wSide = (int) ((W - 4 * inset) / (SECOND_CARD_RT * 2 + 1));
                
                int side = Math.min(wSide, hSide);

                int mediumX = (int) (rawW / 2 - side / 2);
                int leftX = (int) (mediumX - inset - side * SECOND_CARD_RT);
                int rightX = (int) (mediumX + side + inset);
                
                int topY = (int) ((rawH - btnH) / 2 - side / 2);
                int bottomY = (int) (topY + side * (1 - SECOND_CARD_RT) / 2);
                
                prevModelPanel.setBounds(leftX, bottomY, (int) Math.max(1, side * SECOND_CARD_RT), (int) Math.max(1, side * SECOND_CARD_RT));
                selectedModelPanel.setBounds(mediumX, topY, side, side);
                nextModelPanel.setBounds(rightX, bottomY, (int) Math.max(1, side * SECOND_CARD_RT), (int) Math.max(1, side * SECOND_CARD_RT));
            }
        });
    }
    
    protected void setBorderLabel(String text) {
    	lineBorder.setTitle(text);
    	this.validate();
    	this.repaint();
    }
    
    protected void setArrowsEnabled(boolean enabled) {
    	nextButton.setEnabled(enabled);
    	prevButton.setEnabled(enabled);
    }
    
    protected void setLocalBorder() {
    	setBorderLabel(Gui.LOCAL_STR);
    }

    
    protected void setBMZBorder() {
    	setBorderLabel(Gui.BIOIMAGEIO_STR);
    }
    
    protected void setLoading() {
    	defaultString = Gui.LOADING_STR;
    }
    
    protected void setNotFound() {
    	defaultString = Gui.NOT_FOUND_STR;
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            // 1) Create the frame
            JFrame frame = new JFrame("CArrousel Test");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(300, 400);  // or whatever size you need
            frame.setLocationRelativeTo(null);
            GuiAdapter adapter = new GuiAdapter () {

				@Override
				public String getSoftwareName() {
					return "JOHN DOE";
				}
				
				@Override
				public Color getHeaderColor() {
					return Color.gray;
				}

				@Override
				public String getSoftwareDescription() {
					return "The best AI software";
				}

				@Override
				public String getIconPath() {
					return "/home/carlos/git/deep-icy/src/main/resources/deepicy_imgs/icy_logo.png";
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
					return null;
				}

				@Override
				public RunnerAdapter createRunner(ModelDescriptor descriptor, String enginesPath)
						throws IOException, LoadEngineException {
					return null;
				}

				@Override
				public <T extends RealType<T> & NativeType<T>> void displayRai(RandomAccessibleInterval<T> rai,
						String axesOrder, String imTitle) {
					
				}

				@Override
				public <T extends RealType<T> & NativeType<T>> List<Tensor<T>> getInputTensors(
						ModelDescriptor descriptor) {
					return null;
				}

				@Override
				public List<String> getInputImageNames() {
					return null;
				}

				@Override
				public <T extends RealType<T> & NativeType<T>> List<Tensor<T>> convertToInputTensors(
						Map<String, Object> inputs, ModelDescriptor descriptor) {
					return null;
				}

				@Override
				public Color getTitleColor() {
					return Color.white;
				}

				@Override
				public Color getSubtitleColor() {
					return Color.white;
				}
            	
            };
            // 2) Create and configure your card
            ModelSelectionPanelGui card = new ModelSelectionPanelGui(adapter);

            // 3) Add to frame (since ModelCardGui uses null layout internally,
            //    we’ll use BorderLayout here to have it fill the window)
            frame.getContentPane().setLayout(new BorderLayout());
            frame.getContentPane().add(card, BorderLayout.CENTER);

            // 4) Show it
            frame.setVisible(true);
        });
    }
}
