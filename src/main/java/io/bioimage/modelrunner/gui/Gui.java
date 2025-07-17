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

import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.engine.installation.EngineInstall;
import io.bioimage.modelrunner.gui.workers.InstallEnvWorker;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.net.URL;
import java.nio.file.Paths;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.border.EmptyBorder;

import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.adapter.RunnerAdapter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.function.Consumer;

public class Gui extends JPanel {

    private static final long serialVersionUID = 1081914206026104187L;
    private RunnerAdapter runner;
    private GuiAdapter guiAdapter;
	private int currentIndex = 1;
	private final String modelsDir;
	private final String enginesDir;
    private final Object lock = new Object();
    private int nParsedModels;
    private Runnable cancelCallback;
    private boolean cancelled = false;

	Thread engineInstallThread;
	Thread trackEngineInstallThread;
	Thread dwnlThread;
	Thread runninThread;
	Thread finderThread;
	Thread updaterThread;
	Thread localModelsThread;

    private SearchBar searchBar;
    private ContentPanel contentPanel;
    private ModelSelectionPanel modelSelectionPanel;
    private Header titlePanel;
    private JPanel footerPanel;
    private JButton runButton;
    private JButton runOnTestButton;
    private JButton cancelButton;

    protected static final String LOADING_STR = "loading...";
    protected static final String NOT_FOUND_STR = "no models found";
    protected static final String LOCAL_STR = "Local";
    protected static final String BIOIMAGEIO_STR = "Bioimage.io";
    protected static final String RUN_STR = "Run";
    protected static final String CANCEL_STR = "Cancel";
    protected static final String RUN_ON_TEST_STR = "Run on test";
    protected static final String INSTALL_STR = "Install model";
    private static final String MODELS_DEAFULT = "models";
    private static final String ENGINES_DEAFULT = "engines";
    
    protected static final List<String> UNSUPPORTED_MODELS = Arrays.asList(
            "idealistic-rat",
            "diplomatic-bug", "resourceful-lizard", "famous-fish", "happy-elephant",
            "affectionate-cow", "faithful-chicken", "humorous-crab", "noisy-ox",
            "greedy-whale", "efficient-chipmunk",
    		// TODO check cellpose 3d (philosophical-panda)
            "philosophical-panda", "amiable-crocodile",
            // TODO fix the HPA models
            "loyal-parrot", "conscientious-seashell", "straightforward-crocodile", "polite-pig"
        );
    
    private final static String INSTALL_INSTRUCTIONS_FORMAT = ""
    		+ "No models found at: %s" + File.separator + "models<br><br>"
    		+ "Please, install manually or download models from the Bioimage.io.<br><br>"
    		+ "To download models from the Bioimage.io, click on the Bioimage.io button on the top right.";
    
    public static String INSTALL_INSTRUCTIONS = ""
    		+ "No models found.<br><br>"
    		+ "Please, install manually or download models from the Bioimage.io.<br><br>"
    		+ "To download models from the Bioimage.io, click on the Bioimage.io button on the top right.";

    public Gui(GuiAdapter guiAdapter) {
    	DefaultIcon.setIconPath(guiAdapter.getIconPath());
    	INSTALL_INSTRUCTIONS = String.format(INSTALL_INSTRUCTIONS_FORMAT, guiAdapter.getSoftwareName());
        this.guiAdapter = guiAdapter;
        this.modelsDir = guiAdapter.getModelsDir() != null ? guiAdapter.getModelsDir() : new File(MODELS_DEAFULT).getAbsolutePath();
        this.enginesDir = guiAdapter.getEnginesDir() != null ? guiAdapter.getEnginesDir() : new File(ENGINES_DEAFULT).getAbsolutePath();
        loadLocalModels();
        installEnginesIfNeeded();
        setLayout(new GridBagLayout());

        // Initialize UI components
        initTitlePanel();
        initSearchBar();
        initMainContentPanel();
        initFooterPanel();

        setVisible(true);
    }
    
    private void installEnginesIfNeeded() {
    	SwingUtilities.invokeLater(() -> {
    		this.searchBar.switchButton.setEnabled(false);
    		this.runButton.setEnabled(false);
    		this.runOnTestButton.setEnabled(false);
    	});
    	engineInstallThread = new Thread(() -> {
	        EngineInstall installer = EngineInstall.createInstaller(this.enginesDir);
	        installer.checkBasicEngineInstallation();
        	while (titlePanel == null) {
        		try {
					Thread.sleep(30);
				} catch (InterruptedException e) {
					return;
				}
        	}
        	if (installer.getMissingEngines().size() != 0) {
        		titlePanel.setGUIStartInstallation();
    	        installer.setEngineInstalledConsumer(titlePanel.createStringConsumer());
    	        installer.setProgresConsumer(titlePanel.createProgressConsumer());
    	        try {
    				installer.basicEngineInstallation();
    			} catch (InterruptedException e) {
    			}
        	}
	    	SwingUtilities.invokeLater(() -> {
	    		System.out.println("done");
	    		this.searchBar.switchButton.setEnabled(true);
	    		this.runButton.setEnabled(true);
	    		this.runOnTestButton.setEnabled(true);
	    	});
	    });
    	engineInstallThread.start();
    }
    
    private void loadLocalModels() {
	    localModelsThread = new Thread(() -> {
	        List<ModelDescriptor> models = ModelDescriptorFactory.getModelsAtLocalRepo(new File(modelsDir).getAbsolutePath());
	        while (contentPanel == null) {
	        	try {
					Thread.sleep(100);
				} catch (InterruptedException e) {
				}
	        }
            this.setModels(models);
	    });
	    localModelsThread.start();
    }

    private void initTitlePanel() {
    	titlePanel = new Header(this.guiAdapter);
    	titlePanel.setPreferredSize(new Dimension(0, 0));
    	titlePanel.setMinimumSize(new Dimension(0, 0));
    	GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx      = 0;
        gbc.gridwidth  = 1;
        gbc.fill       = GridBagConstraints.BOTH;
        gbc.weightx    = 1.0;
        gbc.gridy     = 0;
        gbc.weighty   = 0.1;   
        this.add(titlePanel, gbc);
    }

    private void initSearchBar() {
        // Set up the title panel
        searchBar = new SearchBar();
        searchBar.setPreferredSize(new Dimension(0, 0));
        searchBar.setMinimumSize(new Dimension(0, 0));
        searchBar.switchButton.addActionListener(ee -> switchBtnClicked());
        searchBar.searchButton.addActionListener(ee -> searchModels());
        searchBar.searchField.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_ENTER)
                	searchModels();
            }
        });
    	GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx      = 0;
        gbc.gridwidth  = 1;
        gbc.fill       = GridBagConstraints.BOTH;
        gbc.weightx    = 1.0;
        gbc.gridy     = 1;
        gbc.weighty   = 0.06;
        this.add(searchBar, gbc);
    }

    private void initMainContentPanel() {
        // Create a main content panel with vertical BoxLayout
        JPanel mainContentPanel = new JPanel(new GridLayout(2, 1));
        mainContentPanel.setPreferredSize(new Dimension(0, 0));
        mainContentPanel.setMinimumSize(new Dimension(0, 0));

        // Add the model selection panel and content panel to the main content panel
        this.modelSelectionPanel = new ModelSelectionPanel(this.guiAdapter);
        mainContentPanel.add(this.modelSelectionPanel);
        contentPanel = new ContentPanel(this.guiAdapter);
        mainContentPanel.add(contentPanel);
        
        modelSelectionPanel.prevButton.addActionListener(e -> updateCarousel(-1));
        modelSelectionPanel.nextButton.addActionListener(e -> updateCarousel(1));
        // Add the main content panel to the frame's CENTER region
    	GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx      = 0;
        gbc.gridwidth  = 1;
        gbc.fill       = GridBagConstraints.BOTH;
        gbc.weightx    = 1.0;
        gbc.gridy     = 2;
        gbc.weighty   = 0.83;   
        add(mainContentPanel, gbc);
    }
    
    private void initFooterPanel() {
        // ───────────────────────────────────────────────────────────────
        // 1) Footer container with GridBagLayout
        footerPanel = new JPanel(new GridBagLayout());
        footerPanel.setPreferredSize(new Dimension(0, 0));
        footerPanel.setMinimumSize(new Dimension(0, 0));
        footerPanel.setBackground(new Color(45, 62, 80));
        footerPanel.setBorder(new EmptyBorder(10, 5, 10, 5));

        // ───────────────────────────────────────────────────────────────
        // 2) Create & style the three buttons
        runOnTestButton = new JButton(RUN_ON_TEST_STR);
        runOnTestButton.addActionListener(e -> runTestOrInstall());
        styleButton(runOnTestButton, "blue");

        runButton = new JButton(RUN_STR);
        runButton.addActionListener(e -> runModel());
        styleButton(runButton, "blue");

        cancelButton = new JButton(CANCEL_STR);
        cancelButton.addActionListener(e -> cancel());
        styleButton(cancelButton, "red");

        // ───────────────────────────────────────────────────────────────
        // 3) runButtonPanel with GridBagLayout so we can weight each button
        JPanel runButtonPanel = new JPanel(new GridBagLayout());
        runButtonPanel.setBackground(new Color(45, 62, 80));
        GridBagConstraints rbGbc = new GridBagConstraints();
        rbGbc.gridy   = 0;
        rbGbc.fill    = GridBagConstraints.BOTH;
        rbGbc.insets  = new Insets(0, 5, 0, 5);

        // Cancel button: weightx = 0 so it's only as wide as it needs to be
        rbGbc.gridx   = 0;
        rbGbc.weightx = 0.2;
        runButtonPanel.add(cancelButton, rbGbc);

        // Run on Test: weightx = 0.5, takes half of remaining space
        rbGbc.gridx   = 1;
        rbGbc.weightx = 0.4;
        runButtonPanel.add(runOnTestButton, rbGbc);

        // Run: weightx = 0.5, takes the other half of remaining space
        rbGbc.gridx   = 2;
        rbGbc.weightx = 0.4;
        runButtonPanel.add(runButton, rbGbc);

        // ───────────────────────────────────────────────────────────────
        // 4) Copyright label
        JLabel copyrightLabel = new JLabel(
            "© 2025 " + guiAdapter.getSoftwareName() + " and JDLL"
        );
        copyrightLabel.setFont(new Font("SansSerif", Font.PLAIN, 12));
        copyrightLabel.setForeground(Color.WHITE);

        // ───────────────────────────────────────────────────────────────
        // 5) Lay out label vs runButtonPanel in footerPanel
        GridBagConstraints fGbc = new GridBagConstraints();
        fGbc.gridy  = 0;
        fGbc.fill   = GridBagConstraints.BOTH;
        fGbc.insets = new Insets(0, 0, 0, 0);

        // Column 0: label takes 40% of width
        fGbc.gridx   = 0;
        fGbc.weightx = 0.4;
        footerPanel.add(copyrightLabel, fGbc);

        // Column 1: runButtonPanel takes 60% of width
        fGbc.gridx   = 1;
        fGbc.weightx = 0.6;
        footerPanel.add(runButtonPanel, fGbc);

        // ───────────────────────────────────────────────────────────────
        // 6) Finally, add footerPanel into your main GridBagLayout at row 3
        GridBagConstraints mainGbc = new GridBagConstraints();
        mainGbc.gridx     = 0;
        mainGbc.gridy     = 3;
        mainGbc.gridwidth = 1;
        mainGbc.fill      = GridBagConstraints.BOTH;
        mainGbc.weightx   = 1.0;
        mainGbc.weighty   = 0.06;
        this.add(footerPanel, mainGbc);
    }


    
    private void cancel() {
    	cancelled = true;
    	if (cancelCallback != null)
    		cancelCallback.run();
    	this.onClose();
    }
    
    private void runTestOrInstall() {
    	if (this.runOnTestButton.getText().equals(INSTALL_STR)) {
    		installSelectedModel();
    	} else if (this.runOnTestButton.getText().equals(RUN_ON_TEST_STR)) {
    		runModelOnTestImage();
    	}
    }
    
    private <T extends RealType<T> & NativeType<T>> void runModel() {
		startModelInstallation(true, "Preparing...");
    	runninThread = new Thread(() -> {
        	try {
        		ModelDescriptor model = modelSelectionPanel.getModels().get(currentIndex);
        		guiAdapter.notifyModelUsed(model.getNickname());
            	if (runner == null || runner.isClosed()) {
                	SwingUtilities.invokeLater(() -> this.contentPanel.setProgressLabelText("Checking deps..."));
                	if (!installEnvToRun(model) && !model.getModelFamily().equals(ModelDescriptor.STARDIST)) {
                		startModelInstallation(false);
            			return;
            		}
                	SwingUtilities.invokeLater(() -> this.contentPanel.setProgressLabelText("Loading model..."));
            		runner = guiAdapter.createRunner(model, this.enginesDir);
            	}
        		if (!runner.isLoaded() && GuiUtils.isEDTAlive())
        			runner.load(false);
        		else if (!GuiUtils.isEDTAlive())
        			return;
            	SwingUtilities.invokeLater(() -> this.contentPanel.setProgressLabelText("Running the model..."));
            	List<String> inputNames = guiAdapter.getInputImageNames();
            	if (inputNames.size() == 0) {
            		startModelInstallation(false);
                	SwingUtilities.invokeLater(() -> this.contentPanel.setProgressLabelText("No image open"));
        			return;
            	}
            	List<Tensor<T>> list = guiAdapter.getInputTensors(runner.getDescriptor());
    			List<Tensor<T>> outs = runner.run(list);
    			for (Tensor<T> tt : outs) {
    				if (!GuiUtils.isEDTAlive())
            			return;
    				guiAdapter.displayRai(tt.getData(), tt.getAxesOrderString(), tt.getName() + "_of_" + inputNames.get(0));
    			}
        		startModelInstallation(false);
    		} catch (Exception e) {
    			if (cancelled)
    				return;
        		startModelInstallation(false);
            	SwingUtilities.invokeLater(() -> this.contentPanel.setProgressLabelText("Error running the model"));
    			e.printStackTrace();
    		}
    	});
    	runninThread.start();
    }
    
    private <T extends RealType<T> & NativeType<T>> void runModelOnTestImage() {
		startModelInstallation(true, "Preparing...");
    	runninThread = new Thread(() -> {
        	try {
        		ModelDescriptor model = modelSelectionPanel.getModels().get(currentIndex);
            	if (runner == null || runner.isClosed()) {
            		if (!installEnvToRun(model) && !model.getModelFamily().equals(ModelDescriptor.STARDIST)) {
                		startModelInstallation(false);
            			return;
            		}
                	SwingUtilities.invokeLater(() -> this.contentPanel.setProgressLabelText("Loading model..."));
            		runner = guiAdapter.createRunner(model, this.enginesDir);
            	}
        		if (!runner.isLoaded() && GuiUtils.isEDTAlive())
        			runner.load(false);
        		else if (!GuiUtils.isEDTAlive())
        			return;
            	SwingUtilities.invokeLater(() -> this.contentPanel.setProgressLabelText("Running the model..."));
    			List<Tensor<T>> outs = runner.runOnTestImages();
            	List<String> inputNames = guiAdapter.getInputImageNames();
    			for (Tensor<T> tt : outs) {
    				if (!GuiUtils.isEDTAlive())
            			return;
    				guiAdapter.displayRai(tt.getData(), tt.getAxesOrderString(), tt.getName() + "_of_" + inputNames.get(0));
    			}
        		startModelInstallation(false);
    		} catch (Exception e) {
    			if (cancelled)
    				return;
        		startModelInstallation(false);
            	SwingUtilities.invokeLater(() -> this.contentPanel.setProgressLabelText("Error running the model"));
    			e.printStackTrace();
    		}
    	});
    	runninThread.start();
    		
    }

    private void updateCarousel(int direction) {
    	closeModelWhenChanging();
    	synchronized(lock) {
            currentIndex = getWrappedIndex(currentIndex + direction);
        	updateProgressBar();

            this.modelSelectionPanel.redrawModelCards(currentIndex);
            
            // Update example image and model info
            int logoHeight = (int) (getHeight() * 0.3);
            int logoWidth = getWidth() / 3;
        	URL coverPath = modelSelectionPanel.getCoverPaths().get(currentIndex);
        	boolean supported = true;
        	if (modelSelectionPanel.getModels().get(currentIndex) != null
        			&& modelSelectionPanel.getModels().get(currentIndex).getModelFamily().equals(ModelDescriptor.BIOIMAGEIO)) {
        		supported = modelSelectionPanel.getModels().get(currentIndex).getWeights().getAllSuportedWeightNames().size() != 0;
	            if (UNSUPPORTED_MODELS.contains(modelSelectionPanel.getModels().get(currentIndex).getNickname()))
	            	supported = false;
        	}
        	contentPanel.setUnsupported(!supported);
        	contentPanel.update(modelSelectionPanel.getModels().get(currentIndex), coverPath, logoWidth, logoHeight);
        	if (this.searchBar.isBarOnLocal()) {
        		this.runOnTestButton.setEnabled(supported);
        		this.runButton.setEnabled(supported);
        	}
    	}
    }
    
    private void updateProgressBar() {
    	if (modelSelectionPanel.getModels().get(currentIndex) == null)
    		return;
    	if (this.searchBar.isBarOnLocal() && this.contentPanel.getProgress() != 0) {
    		contentPanel.setProgressBarText("");
    		contentPanel.setDeterminatePorgress(0);
        	contentPanel.setProgressLabelText("");
    	} else if(!searchBar.isBarOnLocal() && this.contentPanel.getProgress() != 100 
    			&& modelSelectionPanel.getModels().get(currentIndex).isModelInLocalRepo()) {
    		contentPanel.setProgressBarText("100%");
    		contentPanel.setDeterminatePorgress(100);
    	} else if(!searchBar.isBarOnLocal() && this.contentPanel.getProgress() != 0 
    			&& !modelSelectionPanel.getModels().get(currentIndex).isModelInLocalRepo()) {
    		contentPanel.setProgressBarText("");
    		contentPanel.setDeterminatePorgress(0);
        	contentPanel.setProgressLabelText("");
    	}
    	if (searchBar.isBarOnLocal() 
    			|| (!searchBar.isBarOnLocal() 
    					&& !modelSelectionPanel.getModels().get(currentIndex).isModelInLocalRepo() 
    					&& !contentPanel.getProgressBarText().equals(""))) {
    		contentPanel.setProgressBarText("");
        	contentPanel.setProgressLabelText("");
    	}
    }

    private int getWrappedIndex(int index) {
        int size = modelSelectionPanel.getModelNames().size();
        return (index % size + size) % size;
    }

    private void styleButton(JButton button, String color) {
        button.setFont(new Font("SansSerif", Font.BOLD, 14));
        if (color.equals("red")) {
            button.setBackground(new Color(255, 20, 20));
        } else {
            button.setBackground(new Color(52, 152, 219));
        }
        button.setForeground(Color.WHITE);
        button.setFocusPainted(false);
        button.setBorder(BorderFactory.createEmptyBorder(10, 20, 10, 20));
    }
    
    public void setModels(List<ModelDescriptor> models) {
    	if (models.size() == 0)
    		models = createArrayOfNulls(1);
    	this.modelSelectionPanel.setNotFound();
    	this.searchBar.setModels(models);
    	setModelsInGui(models);
    }
    
    protected void setModelsInGui(List<ModelDescriptor> models) {
    	currentIndex = 0;
    	this.modelSelectionPanel.setModels(models);

        // Update example image and model info
        int logoHeight = (int) (getHeight() * 0.3);
        int logoWidth = getWidth() / 3;
    	URL coverPath = modelSelectionPanel.getCoverPaths().get(currentIndex);
    	boolean supported = true;
    	if (modelSelectionPanel.getModels().get(currentIndex) != null
    			&& modelSelectionPanel.getModels().get(currentIndex).getModelFamily().equals(ModelDescriptor.BIOIMAGEIO)) {
    		supported = modelSelectionPanel.getModels().get(currentIndex).getWeights().getAllSuportedWeightNames().size() != 0;
	    	if (UNSUPPORTED_MODELS.contains(modelSelectionPanel.getModels().get(currentIndex).getNickname()))
	        	supported = false;
    	}
        contentPanel.setUnsupported(!supported);
    	contentPanel.update(modelSelectionPanel.getModels().get(currentIndex), coverPath, logoWidth, logoHeight);
    	if (this.searchBar.isBarOnLocal()) {
    		this.runOnTestButton.setEnabled(supported);
    		this.runButton.setEnabled(supported);
    	}
    }
    
    protected void setModelInGuiAt(ModelDescriptor model, int pos) {
    	this.modelSelectionPanel.setModelAt(model, pos);
    	synchronized (lock) {
            if (currentIndex  == pos || currentIndex == pos + 1 || currentIndex == pos - 1
            		|| getWrappedIndex(pos + 1) == currentIndex ) {
            	SwingUtilities.invokeLater(() -> updateCarousel(0));
            }
        }
    }
    
    private void searchModels() {
    	List<ModelDescriptor> models = this.searchBar.performSearch();
    	if (models.size() == 0) {
    		modelSelectionPanel.setNotFound();
    		models = createArrayOfNulls(1);
    	}
    	this.setModelsInGui(models);
    }
    
    protected void switchBtnClicked() {
    	closeModelWhenChanging();
    	if (this.searchBar.isBarOnLocal()) {
    		clickedBMZ();
    	} else {
    		clickedLocal();
    	}
    }
    
    private void closeModelWhenChanging() {
    	if (runner != null && !runner.isClosed()) {
			try {
				runner.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
    	}
    }
    
    
    protected void clickedBMZ() {
    	ArrayList<ModelDescriptor> newModels = createArrayOfNulls(3);
    	this.searchBar.setBarEnabled(false);
    	this.searchBar.changeButtonToLocal();
    	this.contentPanel.setProgressIndeterminate(true);
    	this.contentPanel.setProgressBarText("");
    	this.runButton.setVisible(false);
    	this.runOnTestButton.setText(INSTALL_STR);
    	this.runOnTestButton.setEnabled(false);
    	this.modelSelectionPanel.setBMZBorder();
    	this.contentPanel.setProgressLabelText("Looking for models at Bioimage.io");
    	setModelsInGui(newModels);
    	List<ModelDescriptor> oldModels = new ArrayList<>(searchBar.getBMZModels());
    	
    	finderThread = new Thread(() -> {
    		// This line initiates the read of the bioimage.io collection
    		try {
	        	searchBar.countBMZModels(true);
	        	this.searchBar.findBMZModels();
			} catch (InterruptedException e) {
				return;
			}
    	});
    	
    	updaterThread = new Thread(() -> {
    		try {
        		nParsedModels = 0;
	    		while (oldModels.equals(searchBar.getBMZModels()) && finderThread.isAlive()) {
						Thread.sleep(100);
	    		}
	    		ArrayList<ModelDescriptor> modelsList = createArrayOfNulls(searchBar.countBMZModels(false));
				if (finderThread.isAlive())
					SwingUtilities.invokeLater(() -> setModelsInGui(modelsList));
	    		while (finderThread.isAlive()) {
					Thread.sleep(100);
	            	List<ModelDescriptor> foundModels = new ArrayList<>(searchBar.getBMZModels());
	            	if (foundModels.size() < nParsedModels + 5)
	            		continue;
	            	for (int i = nParsedModels; i < foundModels.size(); i ++) {
	            		setModelInGuiAt(foundModels.get(i), i);
	            	}
	            	nParsedModels = foundModels.size();
	            	
	    		}
	    		if (Thread.currentThread().isInterrupted())
	    			return;
    		} catch (InterruptedException e) {
				return;
			}
    		
    		
    		
        	List<ModelDescriptor> foundModels = searchBar.getBMZModels();
        	for (int i = nParsedModels; i < foundModels.size(); i ++) {
        		int j = 0 + i;
            	SwingUtilities.invokeLater(() -> setModelInGuiAt(foundModels.get(j), j));
        	}
        	SwingUtilities.invokeLater(() -> {
            	this.contentPanel.setProgressIndeterminate(false);
            	this.searchBar.setBarEnabled(true);
            	this.runOnTestButton.setEnabled(true);
            	this.contentPanel.setProgressLabelText("");
        	});
    	});
    	
    	finderThread.start();
    	updaterThread.start();
    }
    
    private static ArrayList<ModelDescriptor> createArrayOfNulls(int n) {
    	ArrayList<ModelDescriptor> newModels = new ArrayList<ModelDescriptor>();
    	for (int i = 0; i < n; i++)
    		newModels.add(null);
    	return newModels;
    }
    
    protected void clickedLocal() {
    	modelSelectionPanel.setLoading();
    	ArrayList<ModelDescriptor> newModels = createArrayOfNulls(3);
    	this.searchBar.setBarEnabled(false);
    	this.searchBar.changeButtonToBMZ();
    	this.contentPanel.setProgressIndeterminate(true);
    	this.contentPanel.setProgressBarText("");
    	this.contentPanel.setProgressLabelText("Looking for models locally");
    	this.runButton.setVisible(true);
    	this.runButton.setEnabled(false);
    	this.runOnTestButton.setText(RUN_ON_TEST_STR);
    	this.runOnTestButton.setEnabled(false);
    	this.modelSelectionPanel.setLocalBorder();
    	this.modelSelectionPanel.setArrowsEnabled(false);
    	setModelsInGui(newModels);
    	
    	Thread finderThread = new Thread(() -> {
        	this.searchBar.findLocalModels(new File(this.modelsDir).getAbsolutePath());
    	});
    	
    	Thread updaterThread = new Thread(() -> {
    		while (finderThread.isAlive()) {
    			try {
					Thread.sleep(500);
				} catch (InterruptedException e) {
					return;
				}
            	List<ModelDescriptor> foundModels = new ArrayList<>(searchBar.getBMZModels());
            	if (foundModels.size() > 0)
            		SwingUtilities.invokeLater(() -> setModelsInGui(foundModels));
    		}
        	List<ModelDescriptor> foundModels = searchBar.getBMZModels();
        	SwingUtilities.invokeLater(() -> {
        		if (foundModels.size() > 0)
            		setModelsInGui(foundModels);
            	this.contentPanel.setProgressIndeterminate(false);
            	this.contentPanel.setDeterminatePorgress(0);
            	this.contentPanel.setProgressBarText("");
            	this.contentPanel.setProgressLabelText("");
            	this.searchBar.setBarEnabled(true);
            	this.runOnTestButton.setEnabled(!contentPanel.isUnsupported());
            	this.modelSelectionPanel.setArrowsEnabled(true);
            	this.runButton.setEnabled(!contentPanel.isUnsupported());
        	});
    	});
    	
    	finderThread.start();
    	updaterThread.start();
    }
    
    private void startModelInstallation(boolean isStarting) {
    	startModelInstallation(isStarting, "Installing...");
    }
    
    private void startModelInstallation(boolean isStarting, String str) {
    	SwingUtilities.invokeLater(() -> {
        	this.runOnTestButton.setEnabled(!isStarting);
        	this.runButton.setEnabled(!isStarting);
        	this.searchBar.setBarEnabled(!isStarting);
        	this.modelSelectionPanel.setArrowsEnabled(!isStarting);
	    	this.contentPanel.setProgressIndeterminate(isStarting);
        	if (isStarting)
        		this.contentPanel.setProgressLabelText(str);
        	else
		    	this.contentPanel.setProgressLabelText("");
    	});
    }
    
    private void checkModelInstallationFinished(CountDownLatch latch) {
    	if (latch.getCount() == 0)
    		startModelInstallation(false);
    }
    
    private void installSelectedModel() {
    	ModelDescriptor selectedModel = modelSelectionPanel.getModels().get(this.currentIndex);
    	Consumer<Double> progress = (c) -> {
			SwingUtilities.invokeLater(() -> {
				double pr = Math.round(c * 10000) / 100d;
				contentPanel.setDeterminatePorgress((int) pr);
				contentPanel.setProgressBarText("" + pr + "%");
			});
    	};
    	startModelInstallation(true);
    	CountDownLatch latch = new CountDownLatch(2);
		this.dwnlThread = new Thread(() -> {
			try {
				String modelFolder = BioimageioRepo.downloadModel(selectedModel, new File(modelsDir).getAbsolutePath(), progress);
				selectedModel.addModelPath(Paths.get(modelFolder));
			} catch (IOException | InterruptedException e) {
    			if (cancelled)
    				return;
				e.printStackTrace();
			}
			latch.countDown();
			checkModelInstallationFinished(latch);
		});
		dwnlThread.start();
		installEnv(selectedModel, latch);
    }
    
    private void installEnv(ModelDescriptor descriptor, CountDownLatch latch) {
    	String msg = "Installation of Python environments might take up to 20 minutes.";
    	String question = String.format("Install %s Python", descriptor.getModelFamily());
    	if (descriptor.areRequirementsInstalled() || !YesNoDialog.askQuestion(question, msg)) {
			latch.countDown();
			checkModelInstallationFinished(latch);
    		return;
    	}
		JDialog installerFrame = new JDialog();
		installerFrame.setTitle("Installing " + descriptor.getName());
		installerFrame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
    	Consumer<Boolean> callback = (bool) -> {
    		checkModelInstallationFinished(latch);
    		if (installerFrame.isVisible())
    			installerFrame.dispose();
    	};
    	InstallEnvWorker worker = new InstallEnvWorker(descriptor, latch, callback);
		EnvironmentInstaller installerPanel = EnvironmentInstaller.create(worker);
    	worker.execute();
		installerPanel.addToFrame(installerFrame);
    	installerFrame.setSize(600, 300);
    }
    
    private boolean installEnvToRun(ModelDescriptor descriptor) {
    	String msg = "The selected model requries Python to run end to end. "
    			+ "Python installation might take up to 20 minutes depending on your computer";
    	String question = String.format("Install %s Python", descriptor.getModelFamily());
    	if (descriptor.areRequirementsInstalled()) {
    		return true;
    	}
    	if (!YesNoDialog.askQuestion(question, msg)) {
    		return false;
    	}
		JDialog[] installerFrame = new JDialog[1];
		InstallEnvWorker[] worker = new InstallEnvWorker[1];
		EnvironmentInstaller[] installerPanel = new EnvironmentInstaller[1];
    	CountDownLatch latch = new CountDownLatch(1);
    	Consumer<Boolean> callback = (bool) -> {
    		if (installerFrame[0].isVisible())
    			installerFrame[0].dispose();
    	};
		try {
			SwingUtilities.invokeAndWait(() -> {
				installerFrame[0] = new JDialog();
				installerFrame[0].setTitle("Installing " + descriptor.getName());
				installerFrame[0].setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
		    	worker[0] = new InstallEnvWorker(descriptor, latch, callback);
				installerPanel[0] = EnvironmentInstaller.create(worker[0]);
			});
		} catch (InvocationTargetException | InterruptedException e) {
			if (cancelled)
				return false;
			throw new RuntimeException(Types.stackTrace(e));
		}
    	worker[0].execute();
    	SwingUtilities.invokeLater(() -> {
    		installerPanel[0].addToFrame(installerFrame[0]);
        	installerFrame[0].setSize(600, 300);
    	});
    	try {
        	latch.await();
		} catch (InterruptedException e) {
			if (cancelled)
				return false;
			e.printStackTrace();
			return false;
		}
    	return true;
    }
    
    public void onClose() {
    	DefaultIcon.closeThreads();
    	if (dwnlThread != null && this.dwnlThread.isAlive())
    		this.dwnlThread.interrupt();
    	if (engineInstallThread != null && this.engineInstallThread.isAlive())
    		this.engineInstallThread.interrupt();
    	if (trackEngineInstallThread != null && this.trackEngineInstallThread.isAlive())
    		this.trackEngineInstallThread.interrupt();
    	if (finderThread != null && this.finderThread.isAlive())
    		this.finderThread.interrupt();
    	if (updaterThread != null && this.updaterThread.isAlive())
    		this.updaterThread.interrupt();
    	if (runninThread != null && runner != null) {
			try {
				runner.close();
				runner = null;
			} catch (IOException e) {
				e.printStackTrace();
			}
    	}
    	if (runninThread != null && this.runninThread.isAlive())
    		this.runninThread.interrupt();
    }
    
    public void setCancelCallback(Runnable callback) {
    	this.cancelCallback = callback;
    }
}
