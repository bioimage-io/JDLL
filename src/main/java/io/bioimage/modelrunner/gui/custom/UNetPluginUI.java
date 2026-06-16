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
package io.bioimage.modelrunner.gui.custom;

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;
import javax.swing.filechooser.FileNameExtensionFilter;

import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.custom.unet.UnetGUI;
import io.bioimage.modelrunner.gui.custom.unet.UnetModelRegistry;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageFiles;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageSourcePanel;

public class UNetPluginUI extends UnetGUI implements ActionListener {

    private static final long serialVersionUID = -1498287706411357145L;
    private static final String DEFAULT_PREVIEW_MESSAGE = "Preview will appear here";
    private static final String SYSTEM_PREVIEW_PROMPT = "Please select an image/folder from the file system";
    private static final String INVALID_IMAGE_MESSAGE = "Please provide a valid image file";
    private static final String EMPTY_FOLDER_MESSAGE = "Folder does not contain valid images";
    private static final Color PREVIEW_ERROR_COLOR = new Color(210, 40, 40);
    private static final String BACKEND_PENDING_MESSAGE = "UNet backend is not implemented yet.";

    private final ConsumerInterface consumer;
    private File selectedSystemPath;
    private File selectedSystemImageFile;
    private boolean windowCloseHookInstalled;
    private Thread workerThread;

    /**
     * Creates a new UNetPluginUI instance.
     *
     * @param consumer the consumer callback.
     * @param adapter the adapter.
     */
    public UNetPluginUI(ConsumerInterface consumer, GuiAdapter adapter) {
        super(adapter);
        this.consumer = consumer;

        String modelsDir = consumer == null ? null : consumer.getModelsDir();
        LinkedHashMap<String, String> unetModelEntries = UnetModelRegistry.buildModelEntries(modelsDir);
        this.inferencePanel.getModelSelectionPanel().setModels(unetModelEntries);
        this.trainPanel.setBaseModels(unetModelEntries);

        this.inferencePanel.getModelSelectionPanel().getBrowseButton().addActionListener(e -> browseInferenceModel());
        this.inferencePanel.getActionPanel().getRunButton().addActionListener(this);
        this.inferencePanel.getActionPanel().getCancelButton().addActionListener(this);
        this.trainPanel.getTrainActionPanel().getRunButton().addActionListener(this);
        this.trainPanel.getTrainActionPanel().getCancelButton().addActionListener(this);
        installInferenceSourceListeners();

        if (this.consumer == null) {
            return;
        }
        List<JComponent> componentList = new ArrayList<JComponent>();
        this.consumer.setVariableNames(null);
        componentList.add(this.inferencePanel.getModelSelectionPanel().getModelComboBox());
        componentList.add(this.inferencePanel.getImageSourcePanel().getOpenImagesComboBox());
        componentList.add(this.inferencePanel.getImageSourcePanel().getFocusButton());
        componentList.add(this.inferencePanel.getImageDisplayPanel());
        this.consumer.setComponents(componentList);
        this.consumer.updateGUI();
    }

    /**
     * Executes close.
     */
    public void close() {
        if (workerThread != null && workerThread.isAlive()) {
            workerThread.interrupt();
        }
    }

    /**
     * Performs add notify.
     */
    @Override
    public void addNotify() {
        super.addNotify();
        installWindowCloseHook();
    }

    private void installWindowCloseHook() {
        if (windowCloseHookInstalled) {
            return;
        }
        java.awt.Window window = SwingUtilities.getWindowAncestor(this);
        if (window == null) {
            return;
        }
        window.addWindowListener(new WindowAdapter() {
            /**
             * Performs window closing.
             *
             * @param e the e.
             */
            @Override
            public void windowClosing(WindowEvent e) {
                close();
            }

            /**
             * Performs window closed.
             *
             * @param e the e.
             */
            @Override
            public void windowClosed(WindowEvent e) {
                close();
            }
        });
        windowCloseHookInstalled = true;
    }

    private void installInferenceSourceListeners() {
        YoloImageSourcePanel sourcePanel = inferencePanel.getImageSourcePanel();
        sourcePanel.getSystemImagesRadio().addActionListener(e -> showSystemPathPrompt());
        sourcePanel.getOpenImagesRadio().addActionListener(e -> showOpenImageSource());
        sourcePanel.getSystemPathField().addActionListener(e -> updateSystemPathPreviewFromField());
        sourcePanel.getSystemPathField().addFocusListener(new FocusAdapter() {
            /**
             * Performs focus lost.
             *
             * @param e the e.
             */
            @Override
            public void focusLost(FocusEvent e) {
                updateSystemPathPreviewFromField();
            }
        });
        sourcePanel.setSystemPathDropConsumer(file -> updateSystemPathPreview(file));
        sourcePanel.getBrowseButton().addActionListener(e -> browseSystemImagePath());
    }

    private void browseInferenceModel() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
        chooser.setFileFilter(new FileNameExtensionFilter("UNet weights (*.pt, *.pth)", "pt", "pth"));
        if (chooser.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        File selected = chooser.getSelectedFile();
        if (selected == null) {
            return;
        }
        File modelFile = selected.isDirectory() ? UnetModelRegistry.findModelFile(selected) : selected;
        String label = selected.isDirectory() ? "[Custom] " + selected.getName()
                : "[Custom] " + UnetModelRegistry.removeWeightsExtension(selected.getName());
        inferencePanel.getModelSelectionPanel().addOrSelectModel(label,
                modelFile == null ? selected.getAbsolutePath() : modelFile.getAbsolutePath());
    }

    private void showSystemPathPrompt() {
        selectedSystemPath = null;
        selectedSystemImageFile = null;
        inferencePanel.getImageSourcePanel().setSystemPathSelectionConfirmed(false);
        inferencePanel.getImageDisplayPanel().setEmptyMessage(SYSTEM_PREVIEW_PROMPT);
        inferencePanel.getImageDisplayPanel().clearImage();
        inferencePanel.updateImageActionState();
    }

    private void showOpenImageSource() {
        selectedSystemPath = null;
        selectedSystemImageFile = null;
        inferencePanel.getImageSourcePanel().setSystemPathSelectionConfirmed(false);
        inferencePanel.getImageDisplayPanel().setEmptyMessage(DEFAULT_PREVIEW_MESSAGE);
        if (consumer != null) {
            consumer.updateGUI();
        } else {
            inferencePanel.getImageDisplayPanel().clearImage();
        }
        inferencePanel.updateImageActionState();
    }

    private void browseSystemImagePath() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
        chooser.setAcceptAllFileFilterUsed(true);
        int result = chooser.showOpenDialog(this);
        if (result != JFileChooser.APPROVE_OPTION) {
            return;
        }
        File selected = chooser.getSelectedFile();
        inferencePanel.getImageSourcePanel().getSystemImagesRadio().setSelected(true);
        inferencePanel.getImageSourcePanel().getSystemPathField().setText(selected.getAbsolutePath());
        updateSystemPathPreview(selected);
    }

    private void updateSystemPathPreviewFromField() {
        if (!inferencePanel.getImageSourcePanel().getSystemImagesRadio().isSelected()) {
            return;
        }
        String text = inferencePanel.getImageSourcePanel().getSystemPathField().getText();
        if (text == null || text.trim().isEmpty()) {
            showSystemPathPrompt();
            return;
        }
        updateSystemPathPreview(new File(text.trim()));
    }

    private void updateSystemPathPreview(File path) {
        selectedSystemPath = null;
        selectedSystemImageFile = null;
        inferencePanel.getImageSourcePanel().setSystemPathSelectionConfirmed(false);
        if (path == null) {
            showPreviewError(INVALID_IMAGE_MESSAGE);
            inferencePanel.updateImageActionState();
            return;
        }
        if (path.isDirectory()) {
            File previewImage = YoloImageFiles.previewImageInDirectory(path);
            if (previewImage == null) {
                showPreviewError(EMPTY_FOLDER_MESSAGE);
                inferencePanel.updateImageActionState();
                return;
            }
            selectedSystemPath = path;
            showSystemPreviewImage(previewImage);
            return;
        }
        if (!YoloImageFiles.canReadImage(path)) {
            showPreviewError(INVALID_IMAGE_MESSAGE);
            inferencePanel.updateImageActionState();
            return;
        }
        selectedSystemPath = path;
        showSystemPreviewImage(path);
    }

    private void showSystemPreviewImage(File imageFile) {
        try {
            inferencePanel.getImageDisplayPanel().setImageFile(imageFile, imageFile.getName(), true);
            selectedSystemImageFile = imageFile;
            inferencePanel.getImageSourcePanel().setSystemPathSelectionConfirmed(true);
        } catch (IOException e) {
            selectedSystemPath = null;
            selectedSystemImageFile = null;
            inferencePanel.getImageSourcePanel().setSystemPathSelectionConfirmed(false);
            showPreviewError(INVALID_IMAGE_MESSAGE);
        }
        inferencePanel.updateImageActionState();
    }

    private void showPreviewError(String message) {
        inferencePanel.getImageDisplayPanel().setEmptyMessage(message, PREVIEW_ERROR_COLOR);
        inferencePanel.getImageDisplayPanel().clearImage();
    }

    /**
     * Executes action performed.
     *
     * @param e the e parameter.
     */
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == this.inferencePanel.getActionPanel().getRunButton()) {
            showBackendPendingMessage();
        } else if (e.getSource() == this.inferencePanel.getActionPanel().getCancelButton()) {
            close();
        } else if (e.getSource() == this.trainPanel.getTrainActionPanel().getRunButton()) {
            if (this.trainPanel.validateTrainingFields()) {
                showBackendPendingMessage();
            }
        } else if (e.getSource() == this.trainPanel.getTrainActionPanel().getCancelButton()) {
            this.trainPanel.setTrainingRunning(false);
            close();
        }
    }

    private void showBackendPendingMessage() {
        inferencePanel.getLogPanel().appendHtml(BACKEND_PENDING_MESSAGE);
        JOptionPane.showMessageDialog(this, BACKEND_PENDING_MESSAGE, "UNet", JOptionPane.INFORMATION_MESSAGE);
    }

    /**
     * Runs this class from the command line.
     *
     * @param args command-line arguments.
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("UNet Plugin");
            frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            frame.getContentPane().add(new UNetPluginUI(null, null));
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
            frame.setResizable(true);
            frame.setSize(400, 300);
        });
    }
}
