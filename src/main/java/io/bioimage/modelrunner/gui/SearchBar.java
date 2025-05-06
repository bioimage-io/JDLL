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

import java.awt.Color;
import java.awt.Desktop;
import java.awt.Font;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.utils.Constants;

public class SearchBar extends JPanel {
    private static final long serialVersionUID = -1741389221668683293L;
    protected JTextField searchField;
    protected LogoPanel bmzLogo;
	protected JButton searchButton;
    protected JButton switchButton;
    private List<ModelDescriptor> bmzModels;
    private int nModels;
    protected static final String SEARCH_ICON_PATH = "jdll_icons/search_logo.png";
    
    double BUTTONS_PERC = 0.4;
    

    protected SearchBar() {
    	this(null);
    }

    protected SearchBar(GuiAdapter adapter) {
        setLayout(null);
        setBackground(Color.WHITE);
        setBorder(BorderFactory.createLineBorder(Color.LIGHT_GRAY, 1, true));

        // Create the search icon
        URL iconPath = getClass().getClassLoader().getResource(SEARCH_ICON_PATH);
        bmzLogo = new LogoPanel();
        if (adapter != null)
            DefaultIcon.drawImOrLogo(Header.class.getClassLoader().getResource(adapter.getIconPath()), iconPath, bmzLogo);
        else
            DefaultIcon.drawImOrLogo(iconPath, iconPath, bmzLogo);
        bmzLogo.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                try {
                	Desktop.getDesktop().browse(new URL("https://www.bioimage.io").toURI());
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        });
        bmzLogo.setBorder(BorderFactory.createEmptyBorder(0, 0, 0, 0));

        // Create the search field
        searchField = new JTextField();
        searchField.setBorder(null);
        searchField.setFont(new Font("Arial", Font.PLAIN, 14));

        // Create the search button
        searchButton = new JButton("Search");
        searchButton.setBackground(new Color(0, 120, 215));
        searchButton.setForeground(Color.WHITE);
        searchButton.setFocusPainted(false);
        searchButton.setBorder(BorderFactory.createEmptyBorder(5, 10, 5, 10));

        // Create the switch button
        switchButton = new JButton(Gui.BIOIMAGEIO_STR);
        switchButton.setBackground(new Color(255, 140, 0));
        switchButton.setForeground(Color.BLACK);
        switchButton.setFocusPainted(false);
        switchButton.setBorder(BorderFactory.createEmptyBorder(5, 10, 5, 10));
        

        add(bmzLogo);
        add(searchField);
        add(searchButton);
        add(switchButton);
        
        organiseComponents();
    }
    
    private void organiseComponents() {
    	addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                layoutAll();
            }
        });
    }
    
    private void layoutAll() {
        int W = getWidth();
        int H = getHeight();
                
        this.bmzLogo.setBounds(0, 0, H, H);
        int searchW = (int) ((W - H) * (1 - BUTTONS_PERC));  

        this.searchField.setBounds(H, 0, searchW, H);
        int buttonW = (int) (0.5 * (W - H) * BUTTONS_PERC);  
        this.searchButton.setBounds(H + searchW, 0, buttonW, H);
        this.switchButton.setBounds(W - buttonW, 0, buttonW, H);
        
        

        //nameLabel.setFont(nameLabel.getFont().deriveFont(Font.BOLD, (float) (16 * scale)));
        //nicknameLabel.setFont(nicknameLabel.getFont().deriveFont(Font.PLAIN, (float) (14 * scale)));
    }
    

    protected List<ModelDescriptor> performSearch() {
        String searchText = searchField.getText().trim().toLowerCase();
        return this.bmzModels.stream().filter(mm -> {
        	if (mm == null) return false;
        	return mm.getName().toLowerCase().contains(searchText) || mm.getDescription().toLowerCase().contains(searchText)
        			|| mm.getNickname().toLowerCase().contains(searchText) || mm.getTags().contains(searchText);
        }).collect(Collectors.toList());
    }
    
    protected int countBMZModels() throws InterruptedException {
    	return countBMZModels(false);
    }
    
    protected int countBMZModels(boolean recount) throws InterruptedException {
    	if (!recount)
    		return nModels;
    	BioimageioRepo.connect();
    	nModels = BioimageioRepo.getModelIDs().size();
    	return nModels;
    }
    
    protected List<ModelDescriptor> findBMZModels() throws InterruptedException {
    	bmzModels = new ArrayList<ModelDescriptor>();
    	for (String url : BioimageioRepo.getModelIDs()) {
    		ModelDescriptor descriptor = BioimageioRepo.retreiveDescriptorFromURL(BioimageioRepo.getModelURL(url) + Constants.RDF_FNAME);
    		if (descriptor == null)
    			continue;
    		bmzModels.add(descriptor);
    	}
    	return bmzModels;
    }
    
    protected void findLocalModels(String dir) {
    	bmzModels = ModelDescriptorFactory.getModelsAtLocalRepo(dir);
    }
    
    protected List<ModelDescriptor> getBMZModels() {
    	return this.bmzModels;
    }
    
    protected void setModels(List<ModelDescriptor> models) {
    	this.bmzModels = models;
    }
    
    protected boolean isBMZPArsingDone() {
    	return nModels == bmzModels.size();
    }
    
    protected void changeButtonToLocal() {
    	this.switchButton.setText(Gui.LOCAL_STR);
    }
    
    protected void changeButtonToBMZ() {
    	this.switchButton.setText(Gui.BIOIMAGEIO_STR);
    }
    
    protected boolean isBarOnLocal() {
    	return this.switchButton.getText().equals(Gui.BIOIMAGEIO_STR);
    }
    
    protected void setBarEnabled(boolean enabled) {
    	this.searchButton.setEnabled(enabled);
    	this.switchButton.setEnabled(enabled);
    	this.searchField.setEnabled(enabled);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                JFrame frame = new JFrame("Modern Search Bar");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.getContentPane().add(new SearchBar());
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
            }
        });
    }
}