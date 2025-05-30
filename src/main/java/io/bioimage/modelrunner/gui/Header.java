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
import java.io.File;
import java.net.URL;
import java.util.function.Consumer;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.border.LineBorder;

import io.bioimage.modelrunner.gui.adapter.GuiAdapter;

public class Header extends JPanel {
    private static final long serialVersionUID = -7691139174208436363L;


    // pulled out so we can resize them on‐the‐fly:
    private final JLabel titleLabel;
    private final JLabel subtitleLabel;

    private JProgressBar progressBar;
    private JLabel progressLabel;

    public Header(GuiAdapter adapter) {
        super(new BorderLayout());
        setForeground(adapter.getHeaderColor());
        setBorder(new LineBorder(Color.BLACK, 2, true));

        // — create + style the two text labels
        titleLabel    = new JLabel(adapter.getSoftwareName(), SwingConstants.CENTER);
        titleLabel.setForeground(adapter.getTitleColor());

        subtitleLabel = new JLabel(adapter.getSoftwareDescription(), SwingConstants.CENTER);
        subtitleLabel.setForeground(adapter.getSubtitleColor());
        
        createProgressBar();
        URL defaultIconUrl = Header.class.getClassLoader().getResource(adapter.getIconPath());
        this.add(new HeaderGui(titleLabel, subtitleLabel, progressBar, progressLabel, defaultIconUrl), BorderLayout.CENTER);
    }

    private void createProgressBar() {
        progressBar = new JProgressBar(0, 100);
        progressBar.setStringPainted(false);
        progressBar.setBackground(Color.LIGHT_GRAY);
        progressBar.setForeground(new Color(46, 204, 113));

        progressLabel = new JLabel("Processing...", SwingConstants.CENTER);
        progressLabel.setBackground(Color.GRAY);
        progressLabel.setOpaque(true);
    }
	
	protected void setGUIStartInstallation() {
    	SwingUtilities.invokeLater(() -> {
    		progressBar.setIndeterminate(true);
            progressBar.setStringPainted(true);
    		progressBar.setVisible(true);
    		progressBar.setString("0%");
    		progressLabel.setText("Preparing installation...");
    		progressLabel.setVisible(true);

    	});
	}
	
	public Consumer<String> createStringConsumer(){
		Consumer<String> consumer = (ss) -> {
			SwingUtilities.invokeLater(() -> {
    			this.progressLabel.setText("Installing " + new File(ss).getName());
			});
		};
		return consumer;
	}
	
	public Consumer<Double> createProgressConsumer(){
		Consumer<Double> consumer = (dd) -> {
			SwingUtilities.invokeLater(() -> {
				if (progressBar.isIndeterminate())
					progressBar.setIndeterminate(false);
				double perc = Math.floor(dd * 1000) / 10;
				progressBar.setString(perc + "%");
	    		progressBar.setValue((int) perc);
	    		if (perc == 100){
	        		progressLabel.setVisible(false);
	        		progressBar.setVisible(false);
	    		}
			});
		};
		return consumer;
	}
}
