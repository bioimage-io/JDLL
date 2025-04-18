package io.bioimage.modelrunner.gui;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.engine.installation.EngineInstall;
import io.bioimage.modelrunner.gui.workers.InstallEnvWorker;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.io.File;
import java.io.IOException;
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
    private Layout layout = Layout.createVertical(LAYOUT_WEIGHTS);

    private static final double FOOTER_VRATIO = 0.06;
    private static final double[] LAYOUT_WEIGHTS = new double[] {0.1, 0.05, 0.8, 0.05};

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
        long tt = System.currentTimeMillis();
        this.modelsDir = guiAdapter.getModelsDir() != null ? guiAdapter.getModelsDir() : new File(MODELS_DEAFULT).getAbsolutePath();
        this.enginesDir = guiAdapter.getEnginesDir() != null ? guiAdapter.getEnginesDir() : new File(ENGINES_DEAFULT).getAbsolutePath();
        loadLocalModels();
        System.out.println("Model loading: " + (System.currentTimeMillis() - tt));
        tt = System.currentTimeMillis();
        installEnginesIfNeeded();
        System.out.println("Engines loading: " + (System.currentTimeMillis() - tt));
        tt = System.currentTimeMillis();
        setSize(800, 900);
        setLayout(layout);
        System.out.println("Set size: " + (System.currentTimeMillis() - tt));
        tt = System.currentTimeMillis();

        // Initialize UI components
        initTitlePanel();
        System.out.println("Title panel: " + (System.currentTimeMillis() - tt));
        tt = System.currentTimeMillis();
        initSearchBar();
        System.out.println("Search bar: " + (System.currentTimeMillis() - tt));
        tt = System.currentTimeMillis();
        initMainContentPanel();
        System.out.println("Content panel: " + (System.currentTimeMillis() - tt));
        tt = System.currentTimeMillis();
        initFooterPanel();
        System.out.println("Footer: " + (System.currentTimeMillis() - tt));

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
    	titlePanel = new Header(this.guiAdapter.getSoftwareName(), this.guiAdapter.getSoftwareDescription(), this.getWidth(), this.getHeight());
        add(titlePanel, layout.get(0));
    }

    private void initSearchBar() {
        // Set up the title panel
        searchBar = new SearchBar(this.getWidth(), this.getHeight());
        add(searchBar, layout.get(1));
        searchBar.switchButton.addActionListener(ee -> switchBtnClicked());
        searchBar.searchButton.addActionListener(ee -> searchModels());
        searchBar.searchField.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_ENTER)
                	searchModels();
            }
        });
    }

    private void initMainContentPanel() {
        // Create a main content panel with vertical BoxLayout
        JPanel mainContentPanel = new JPanel();
        Layout mainPanelLayout = Layout.createVertical(new double[] {0.45, 0.55});
        mainContentPanel.setLayout(mainPanelLayout);
        mainContentPanel.setBackground(Color.WHITE);

        // Add the model selection panel and content panel to the main content panel
        this.modelSelectionPanel = new ModelSelectionPanel(this.getWidth(), this.getHeight());
        mainContentPanel.add(this.modelSelectionPanel, mainPanelLayout.get(0));
        contentPanel = new ContentPanel(this.getWidth(), this.getHeight());
        mainContentPanel.add(contentPanel, mainPanelLayout.get(1));

        // Add the main content panel to the frame's CENTER region
        add(mainContentPanel, layout.get(2));
        
        modelSelectionPanel.prevButton.addActionListener(e -> updateCarousel(-1));
        modelSelectionPanel.nextButton.addActionListener(e -> updateCarousel(1));
    }

    private void initFooterPanel() {
        footerPanel = new JPanel(new BorderLayout());
        footerPanel.setBackground(new Color(45, 62, 80));
        footerPanel.setBorder(new EmptyBorder(10, 5, 10, 5));
        footerPanel.setPreferredSize(new Dimension(this.getWidth(), (int) (this.getHeight() * FOOTER_VRATIO)));

        JPanel runButtonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 20, 0));
        runButtonPanel.setBackground(new Color(45, 62, 80));

        runOnTestButton = new JButton(RUN_ON_TEST_STR);
        runOnTestButton.addActionListener(e -> runTestOrInstall());
        runButton = new JButton(RUN_STR);
        runButton.addActionListener(e -> runModel());
        cancelButton = new JButton(CANCEL_STR);
        cancelButton.addActionListener(e -> cancel());

        styleButton(runOnTestButton, "blue");
        styleButton(runButton, "blue");
        styleButton(cancelButton, "red");

        runButtonPanel.add(cancelButton);
        runButtonPanel.add(runOnTestButton);
        runButtonPanel.add(runButton);

        JLabel copyrightLabel = new JLabel("© 2024 " + guiAdapter.getSoftwareName() + " and JDLL");
        copyrightLabel.setFont(new Font("SansSerif", Font.PLAIN, 12));
        copyrightLabel.setForeground(Color.WHITE);

        footerPanel.add(runButtonPanel, BorderLayout.EAST);
        footerPanel.add(copyrightLabel, BorderLayout.WEST);

        add(footerPanel, layout.get(3));
    }
    
    private void cancel() {
    	this.onClose();
    }
    
    private <T extends RealType<T> & NativeType<T>> void runModel() {
    	SwingUtilities.invokeLater(() -> this.contentPanel.setProgressIndeterminate(true));
    	runninThread = new Thread(() -> {
        	try {
            	if (runner == null || runner.isClosed()) {
                	SwingUtilities.invokeLater(() -> this.contentPanel.setProgressLabelText("Loading model..."));
            		runner = guiAdapter.createRunner(this.modelSelectionPanel.getModels().get(currentIndex));
            	}
        		if (!runner.isLoaded() && GuiUtils.isEDTAlive())
        			runner.load();
        		else if (!GuiUtils.isEDTAlive())
        			return;
            	SwingUtilities.invokeLater(() -> this.contentPanel.setProgressLabelText("Running the model..."));
            	List<String> inputNames = guiAdapter.getInputImageNames();
            	List<Tensor<T>> list = guiAdapter.getInputTensors(runner.getDescriptor());
    			List<Tensor<T>> outs = runner.run(list);
    			for (Tensor<T> tt : outs) {
    				if (!GuiUtils.isEDTAlive())
            			return;
    				if (!GuiUtils.isEDTAlive())
            			return;
    				guiAdapter.displayRai(tt.getData(), tt.getAxesOrderString(), tt.getName() + "_of_" + inputNames.get(0));
    			}
    		} catch (Exception e) {
    			e.printStackTrace();
    		}
        	SwingUtilities.invokeLater(() -> {
        		this.contentPanel.setProgressLabelText("");
        		this.contentPanel.setProgressIndeterminate(false);
        	});
    	});
    	runninThread.start();
    }
    
    private void runTestOrInstall() {
    	if (this.runOnTestButton.getText().equals(INSTALL_STR)) {
    		installSelectedModel();
    	} else if (this.runOnTestButton.getText().equals(RUN_ON_TEST_STR)) {
    		runModelOnTestImage();
    	}
    }
    
    private <T extends RealType<T> & NativeType<T>> void runModelOnTestImage() {
    	SwingUtilities.invokeLater(() -> this.contentPanel.setProgressIndeterminate(true));
    	runninThread = new Thread(() -> {
        	try {
            	if (runner == null || runner.isClosed()) {
                	SwingUtilities.invokeLater(() -> this.contentPanel.setProgressLabelText("Loading model..."));
            		runner = guiAdapter.createRunner(this.modelSelectionPanel.getModels().get(currentIndex));
            	}
        		if (!runner.isLoaded() && GuiUtils.isEDTAlive())
        			runner.load();
        		else if (!GuiUtils.isEDTAlive())
        			return;
            	SwingUtilities.invokeLater(() -> this.contentPanel.setProgressLabelText("Running the model..."));
    			List<Tensor<T>> outs = runner.runOnTestImages();
            	List<String> inputNames = guiAdapter.getInputImageNames();
    			for (Tensor<T> tt : outs) {
    				if (!GuiUtils.isEDTAlive())
            			return;
    				if (!GuiUtils.isEDTAlive())
            			return;
    				guiAdapter.displayRai(tt.getData(), tt.getAxesOrderString(), tt.getName() + "_of_" + inputNames.get(0));
    			}
    		} catch (Exception e) {
    			e.printStackTrace();
    		}
        	SwingUtilities.invokeLater(() -> {
        		this.contentPanel.setProgressLabelText("");
        		this.contentPanel.setProgressIndeterminate(false);
        	});
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
            contentPanel.update(modelSelectionPanel.getModels().get(currentIndex), coverPath, logoWidth, logoHeight);
    	}
    }
    
    private void updateProgressBar() {
    	if (modelSelectionPanel.getModels().get(currentIndex) == null)
    		return;
    	if (this.searchBar.isBarOnLocal() && this.contentPanel.getProgress() != 0) {
    		contentPanel.setProgressBarText("");
    		contentPanel.setDeterminatePorgress(0);
    	} else if(!searchBar.isBarOnLocal() && this.contentPanel.getProgress() != 100 
    			&& modelSelectionPanel.getModels().get(currentIndex).isModelInLocalRepo()) {
    		contentPanel.setProgressBarText("100%");
    		contentPanel.setDeterminatePorgress(100);
    	} else if(!searchBar.isBarOnLocal() && this.contentPanel.getProgress() != 0 
    			&& !modelSelectionPanel.getModels().get(currentIndex).isModelInLocalRepo()) {
    		contentPanel.setProgressBarText("");
    		contentPanel.setDeterminatePorgress(0);
    	}
    	if (searchBar.isBarOnLocal() 
    			|| (!searchBar.isBarOnLocal() 
    					&& !modelSelectionPanel.getModels().get(currentIndex).isModelInLocalRepo() 
    					&& !contentPanel.getProgressBarText().equals(""))) {
    		contentPanel.setProgressBarText("");
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
        contentPanel.update(modelSelectionPanel.getModels().get(currentIndex), coverPath, logoWidth, logoHeight);
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
            	this.runOnTestButton.setEnabled(true);
            	this.modelSelectionPanel.setArrowsEnabled(true);
            	this.runButton.setEnabled(true);
        	});
    	});
    	
    	finderThread.start();
    	updaterThread.start();
    }
    
    private void startModelInstallation(boolean isStarting) {
    	SwingUtilities.invokeLater(() -> {
        	this.runOnTestButton.setEnabled(!isStarting);
        	this.searchBar.setBarEnabled(!isStarting);
        	this.modelSelectionPanel.setArrowsEnabled(!isStarting);
        	if (isStarting)
        		this.contentPanel.setProgressLabelText("Installing ...");
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
			SwingUtilities.invokeLater(() -> contentPanel.setDeterminatePorgress((int) (c * 100)));
    	};
    	startModelInstallation(true);
    	CountDownLatch latch = new CountDownLatch(2);
		this.dwnlThread = new Thread(() -> {
			try {
				String modelFolder = BioimageioRepo.downloadModel(selectedModel, new File(modelsDir).getAbsolutePath(), progress);
				selectedModel.addModelPath(Paths.get(modelFolder));
			} catch (IOException | InterruptedException e) {
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
    	Runnable callback = () -> {
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
}
