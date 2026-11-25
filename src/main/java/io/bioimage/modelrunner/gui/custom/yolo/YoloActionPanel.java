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

import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.util.Arrays;
import java.util.List;
import java.util.prefs.Preferences;

import javax.swing.JButton;
import javax.swing.JPanel;
import io.bioimage.modelrunner.gui.custom.segmentation.SegmentationOutput;

public class YoloActionPanel extends JPanel {

    private static final long serialVersionUID = -763353214342801966L;

    private static final int GAP = 6;
    private static final List<SegmentationOutput> OUTPUTS = Arrays.asList(
            SegmentationOutput.LABELS, SegmentationOutput.BOTH, SegmentationOutput.ROIS);

    protected final JButton cancelButton = new JButton("Cancel");
    protected final JButton runButton = new JButton("Infer");
    private SegmentationActionCarousel carousel;
    private YoloImageSourcePanel segmentationSource;
    private boolean openRois = true, sourceReady = true, outputLocked;
    private SegmentationOutput openOutput, fileOutput;
    private Preferences preferences;

    /** Only final-label workflows enable the three-way action. */
    public void enableSegmentationOutputs(YoloImageSourcePanel source) {
        enableSegmentationOutputs(source, Preferences.userNodeForPackage(YoloActionPanel.class));
    }

    void enableSegmentationOutputs(YoloImageSourcePanel source, Preferences settings) {
        segmentationSource = source;
        preferences = settings;
        openOutput = preference("open");
        fileOutput = preference("files");
        carousel = new SegmentationActionCarousel(runButton, OUTPUTS, this::browse);
        add(carousel);
        source.getSystemImagesRadio().addItemListener(e -> updateOutput());
        updateOutput();
    }

    public void setOpenImageRoisSupported(boolean supported) {
        openRois = supported;
        if (!supported) openOutput = SegmentationOutput.LABELS;
        updateOutput();
    }

    private SegmentationOutput preference(String source) {
        SegmentationOutput fallback = SegmentationOutput.BOTH;
        try { return SegmentationOutput.valueOf(preferences.get("segmentation." + source, fallback.name())); }
        catch (IllegalArgumentException e) { return fallback; }
    }

    private void browse(int direction) {
        boolean files = segmentationSource.getSystemImagesRadio().isSelected();
        int index = Math.floorMod(OUTPUTS.indexOf(output()) + direction, OUTPUTS.size());
        SegmentationOutput value = OUTPUTS.get(index);
        if (files) fileOutput = value; else openOutput = value;
        preferences.put("segmentation." + (files ? "files" : "open"), value.name());
        updateOutput();
    }

    public SegmentationOutput output() {
        if (segmentationSource == null) return SegmentationOutput.LABELS;
        return segmentationSource.getSystemImagesRadio().isSelected() ? fileOutput : openOutput;
    }

    public void setOutputLocked(boolean locked) {
        outputLocked = locked;
        updateOutput();
    }

    void setRunAvailable(boolean available) {
        sourceReady = available;
        if (segmentationSource == null) runButton.setEnabled(available);
        else updateOutput();
    }

    private void updateOutput() {
        if (carousel == null) return;
        SegmentationOutput selection = output();
        boolean supported = openRois || segmentationSource.getSystemImagesRadio().isSelected()
                || selection == SegmentationOutput.LABELS;
        runButton.setEnabled(sourceReady && supported && !outputLocked);
        carousel.select(selection, !outputLocked);
        runButton.setToolTipText(supported ? selection.actionLabel : "Update the host adapter to add ROIs to an open image.");
        doLayout();
    }

    /**
     * Creates a new YoloActionPanel instance.
     */
    protected YoloActionPanel() {
        setLayout(null);
        setOpaque(false);
        YoloUiUtils.styleFlatSecondaryButton(cancelButton);
        YoloUiUtils.styleFlatSecondaryButton(runButton);
        add(cancelButton);
        add(runButton);
        addComponentListener(new ComponentAdapter() {
            /**
             * Performs component resized.
             *
             * @param e the e.
             */
            @Override
            public void componentResized(ComponentEvent e) {
                doLayout();
            }
        });
    }

    /**
     * Performs do layout.
     */
    @Override
    public void doLayout() {
        int w = Math.max(0, getWidth());
        int h = Math.max(0, getHeight());
        int cancelW = Math.max(1, (w - GAP) / (carousel != null ? 3 : 2));
        int runW = Math.max(1, w - GAP - cancelW);
        cancelButton.setBounds(0, 0, cancelW, h);
        if (carousel != null) {
            carousel.setBounds(cancelW + GAP, 0, runW, h);
            carousel.doLayout();
        } else {
            runButton.setBounds(cancelW + GAP, 0, runW, h);
            YoloUiUtils.applyResponsiveText(runButton, runW - 8, h);
        }
        YoloUiUtils.applyResponsiveText(cancelButton, cancelW - 8, h);
    }

    /**
     * Returns the cancel button.
     *
     * @return the cancel button.
     */
    public JButton getCancelButton() {
        return cancelButton;
    }

    /**
     * Returns the run button.
     *
     * @return the run button.
     */
    public JButton getRunButton() {
        return runButton;
    }
}
