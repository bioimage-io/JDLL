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

import java.awt.datatransfer.DataFlavor;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.TransferHandler;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

public class YoloImageSourcePanel extends JPanel {

    private static final long serialVersionUID = 5381451655035760876L;

    private static final int GAP = 6;
    private static final double RADIO_LABEL_GAP_RATIO = 0.015;
    private static final double RADIO_WIDTH_RATIO = 0.05;
    private static final double HELP_WIDTH_RATIO = 0.05;
    private static final double FOCUS_WIDTH_RATIO = 0.12;
    private static final double BROWSE_WIDTH_RATIO = 0.14;
    private static final double NAV_BUTTON_SIZE_RATIO = 0.7125;

    protected final JRadioButton openImagesRadio = new JRadioButton();
    protected final JRadioButton systemImagesRadio = new JRadioButton();

    protected final JComboBox<YoloImageSelectionEntry> openImagesComboBox =
            new JComboBox<YoloImageSelectionEntry>(new YoloOpenImageComboBoxModel());
    protected final JButton previousImageButton = new JButton("<");
    protected final JButton nextImageButton = new JButton(">");
    protected final JButton focusButton = new JButton("Focus");
    protected final YoloHelpIcon openImagesHelpIcon = new YoloHelpIcon();

    protected final YoloPlaceholderTextField systemPathField = new YoloPlaceholderTextField("path to an image or folder");
    protected final JButton browseButton = new JButton("Browse");
    protected final YoloHelpIcon systemPathHelpIcon = new YoloHelpIcon();
    private final FileDropHandler systemPathDropHandler;
    private boolean systemPathSelectionConfirmed;

    /**
     * Creates a new YoloImageSourcePanel instance.
     */
    protected YoloImageSourcePanel() {
        setLayout(null);
        setOpaque(false);
        openImagesRadio.setSelected(true);
        ButtonGroup group = new ButtonGroup();
        group.add(openImagesRadio);
        group.add(systemImagesRadio);

        systemPathDropHandler = new FileDropHandler(systemPathField);
        systemPathField.setTransferHandler(systemPathDropHandler);
        openImagesComboBox.setRenderer(new YoloOpenImageComboBoxRenderer());
        YoloUiUtils.styleInput(openImagesComboBox);
        YoloUiUtils.styleInput(systemPathField);
        YoloUiUtils.styleFlatSecondaryButton(previousImageButton);
        YoloUiUtils.styleFlatSecondaryButton(nextImageButton);
        YoloUiUtils.styleFlatSecondaryButton(focusButton);
        YoloUiUtils.styleFlatSecondaryButton(browseButton);
        openImagesHelpIcon.setToolTipText("Select an already open image, move through the open-image list, or bring the selected image into focus.");
        systemPathHelpIcon.setToolTipText("Provide an image or folder from the file system. Drag and drop is supported.");
        previousImageButton.addActionListener(e -> selectRelativeOpenImage(-1));
        nextImageButton.addActionListener(e -> selectRelativeOpenImage(1));

        add(openImagesRadio);
        add(openImagesComboBox);
        add(previousImageButton);
        add(nextImageButton);
        add(focusButton);
        add(openImagesHelpIcon);

        add(systemImagesRadio);
        add(systemPathField);
        add(browseButton);
        add(systemPathHelpIcon);

        openImagesRadio.addActionListener(e -> updateEnabledState());
        systemImagesRadio.addActionListener(e -> updateEnabledState());
        systemPathField.getDocument().addDocumentListener(new DocumentListener() {
            /**
             * Performs insert update.
             *
             * @param e the e.
             */
            @Override
            public void insertUpdate(DocumentEvent e) {
                invalidateSystemPathSelection();
            }

            /**
             * Performs remove update.
             *
             * @param e the e.
             */
            @Override
            public void removeUpdate(DocumentEvent e) {
                invalidateSystemPathSelection();
            }

            /**
             * Performs changed update.
             *
             * @param e the e.
             */
            @Override
            public void changedUpdate(DocumentEvent e) {
                invalidateSystemPathSelection();
            }
        });
        updateEnabledState();
    }

    private void invalidateSystemPathSelection() {
        systemPathSelectionConfirmed = false;
    }

    private void selectRelativeOpenImage(int step) {
        int count = openImagesComboBox.getItemCount();
        if (count <= 1) {
            return;
        }
        int selectedIndex = openImagesComboBox.getSelectedIndex();
        if (selectedIndex < 0) {
            openImagesComboBox.setSelectedIndex(0);
            return;
        }
        int nextIndex = (selectedIndex + step + count) % count;
        openImagesComboBox.setSelectedIndex(nextIndex);
    }

    /**
     * Performs do layout.
     */
    @Override
    public void doLayout() {
        int w = Math.max(0, getWidth());
        int h = Math.max(0, getHeight());
        int rowH = Math.max(1, (h - GAP) / 2);

        int radioW = (int) Math.round(w * RADIO_WIDTH_RATIO);
        int radioGap = (int) Math.round(w * RADIO_LABEL_GAP_RATIO);
        int helpW = Math.max(12, (int) Math.round(w * HELP_WIDTH_RATIO));
        int navH = Math.max(1, (int) Math.round(rowH * NAV_BUTTON_SIZE_RATIO));
        int navW = navH;
        int focusW = (int) Math.round(w * FOCUS_WIDTH_RATIO);
        int browseW = (int) Math.round(w * BROWSE_WIDTH_RATIO);
        int contentX = radioW + radioGap;
        int openRowY = (rowH - navH) / 2;
        int helpY = (rowH - helpW) / 2;

        int rowY = 0;
        int comboW = Math.max(1, w - contentX - 2 * navW - focusW - helpW - 3 * GAP);
        openImagesRadio.setBounds(0, rowY, radioW, rowH);
        openImagesComboBox.setBounds(contentX, rowY, comboW, rowH);
        previousImageButton.setBounds(contentX + comboW + GAP, openRowY, navW, navH);
        nextImageButton.setBounds(contentX + comboW + GAP + navW, openRowY, navW, navH);
        focusButton.setBounds(contentX + comboW + GAP + 2 * navW + GAP, rowY, focusW, rowH);
        openImagesHelpIcon.setBounds(w - helpW, rowY + helpY, helpW, helpW);

        rowY = rowH + GAP;
        int textW = Math.max(1, w - contentX - browseW - helpW - 2 * GAP);
        systemImagesRadio.setBounds(0, rowY, radioW, rowH);
        systemPathField.setBounds(contentX, rowY, textW, rowH);
        browseButton.setBounds(contentX + textW + GAP, rowY, browseW, rowH);
        systemPathHelpIcon.setBounds(w - helpW, rowY + helpY, helpW, helpW);

        YoloUiUtils.applyResponsiveFont(openImagesComboBox, rowH);
        YoloUiUtils.applyResponsiveFont(systemPathField, rowH);
        YoloUiUtils.applyResponsiveText(previousImageButton, navW - 4, navH);
        YoloUiUtils.applyResponsiveText(nextImageButton, navW - 4, navH);
        YoloUiUtils.applyResponsiveText(focusButton, focusW - 8, rowH);
        YoloUiUtils.applyResponsiveText(browseButton, browseW - 8, rowH);
        openImagesRadio.setFont(focusButton.getFont());
        systemImagesRadio.setFont(browseButton.getFont());
        openImagesRadio.setOpaque(false);
        systemImagesRadio.setOpaque(false);
    }

    /**
     * Sets the open images.
     *
     * @param entries the entries.
     */
    public void setOpenImages(List<YoloImageSelectionEntry> entries) {
        ((YoloOpenImageComboBoxModel) openImagesComboBox.getModel()).setEntries(entries);
    }

    /**
     * Sets the open image titles.
     *
     * @param names the names.
     */
    public void setOpenImageTitles(List<String> names) {
        List<YoloImageSelectionEntry> entries = new ArrayList<YoloImageSelectionEntry>();
        for (int i = 0; i < names.size(); i++) {
            entries.add(new YoloImageSelectionEntry(Integer.toString(i), names.get(i), null));
        }
        setOpenImages(entries);
    }

    /**
     * Performs update enabled state.
     */
    public void updateEnabledState() {
        boolean openSelected = openImagesRadio.isSelected();
        boolean hasOpenImage = hasValidOpenImageSelection();
        boolean hasMultipleOpenImages = openImagesComboBox.getItemCount() > 1;
        openImagesComboBox.setEnabled(openSelected);
        previousImageButton.setEnabled(openSelected && hasOpenImage && hasMultipleOpenImages);
        nextImageButton.setEnabled(openSelected && hasOpenImage && hasMultipleOpenImages);
        focusButton.setEnabled(openSelected && hasOpenImage);
        openImagesHelpIcon.setEnabled(openSelected);

        boolean systemSelected = systemImagesRadio.isSelected();
        systemPathField.setEnabled(systemSelected);
        browseButton.setEnabled(systemSelected);
        systemPathHelpIcon.setEnabled(systemSelected);
    }

    /**
     * Returns whether has valid open image selection.
     *
     * @return true if has valid open image selection; false otherwise.
     */
    public boolean hasValidOpenImageSelection() {
        return openImagesComboBox.getSelectedItem() instanceof YoloImageSelectionEntry;
    }

    /**
     * Returns whether has valid system path selection.
     *
     * @return true if has valid system path selection; false otherwise.
     */
    public boolean hasValidSystemPathSelection() {
        if (!systemPathSelectionConfirmed) {
            return false;
        }
        String path = systemPathField.getText();
        if (path == null || path.trim().isEmpty()) {
            return false;
        }
        File file = new File(path.trim());
        return YoloImageFiles.isValidDroppedPath(file);
    }

    /**
     * Returns whether has valid selected source.
     *
     * @return true if has valid selected source; false otherwise.
     */
    public boolean hasValidSelectedSource() {
        return openImagesRadio.isSelected() ? hasValidOpenImageSelection() : hasValidSystemPathSelection();
    }

    /**
     * Returns the open images radio.
     *
     * @return the open images radio.
     */
    public JRadioButton getOpenImagesRadio() {
        return openImagesRadio;
    }

    /**
     * Returns the system images radio.
     *
     * @return the system images radio.
     */
    public JRadioButton getSystemImagesRadio() {
        return systemImagesRadio;
    }

    /**
     * Returns the open images combo box.
     *
     * @return the open images combo box.
     */
    public JComboBox<YoloImageSelectionEntry> getOpenImagesComboBox() {
        return openImagesComboBox;
    }

    /**
     * Returns the system path field.
     *
     * @return the system path field.
     */
    public YoloPlaceholderTextField getSystemPathField() {
        return systemPathField;
    }

    /**
     * Returns the browse button.
     *
     * @return the browse button.
     */
    public JButton getBrowseButton() {
        return browseButton;
    }

    /**
     * Returns the previous image button.
     *
     * @return the previous image button.
     */
    public JButton getPreviousImageButton() {
        return previousImageButton;
    }

    /**
     * Returns the next image button.
     *
     * @return the next image button.
     */
    public JButton getNextImageButton() {
        return nextImageButton;
    }

    /**
     * Returns the focus button.
     *
     * @return the focus button.
     */
    public JButton getFocusButton() {
        return focusButton;
    }

    /**
     * Sets the system path drop consumer.
     *
     * @param dropConsumer the drop consumer callback.
     */
    public void setSystemPathDropConsumer(Consumer<File> dropConsumer) {
        systemPathDropHandler.setDropConsumer(dropConsumer);
    }

    /**
     * Sets the system path selection confirmed.
     *
     * @param confirmed the confirmed.
     */
    public void setSystemPathSelectionConfirmed(boolean confirmed) {
        systemPathSelectionConfirmed = confirmed;
        updateEnabledState();
    }

    private static class FileDropHandler extends TransferHandler {
        private static final long serialVersionUID = -4079793236252082911L;
        private final YoloPlaceholderTextField field;
        private Consumer<File> dropConsumer;

        FileDropHandler(YoloPlaceholderTextField field) {
            this.field = field;
        }

        private void setDropConsumer(Consumer<File> dropConsumer) {
            this.dropConsumer = dropConsumer;
        }

        /**
         * Returns whether can import.
         *
         * @param support the support.
         * @return true if can import; false otherwise.
         */
        @Override
        public boolean canImport(TransferSupport support) {
            return support.isDataFlavorSupported(DataFlavor.javaFileListFlavor);
        }

        /**
         * Returns whether import data.
         *
         * @param support the support.
         * @return true if import data; false otherwise.
         */
        @Override
        public boolean importData(TransferSupport support) {
            try {
                @SuppressWarnings("unchecked")
                List<File> files = (List<File>) support.getTransferable().getTransferData(DataFlavor.javaFileListFlavor);
                if (files == null || files.isEmpty()) {
                    return false;
                }
                File file = files.get(0);
                field.setText(file.getAbsolutePath());
                if (dropConsumer != null) {
                    dropConsumer.accept(file);
                }
                return true;
            } catch (Exception ex) {
                return false;
            }
        }
    }
}
