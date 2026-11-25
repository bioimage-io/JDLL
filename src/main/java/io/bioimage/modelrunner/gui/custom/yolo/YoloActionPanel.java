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

import javax.swing.JButton;
import javax.swing.JPanel;
import java.awt.Insets;
import java.util.prefs.Preferences;
import io.bioimage.modelrunner.gui.custom.segmentation.SegmentationOutput;

public class YoloActionPanel extends JPanel {

    private static final long serialVersionUID = -763353214342801966L;

    private static final int GAP = 6;

    protected final JButton cancelButton = new JButton("Cancel");
    protected final JButton runButton = new JButton("Infer");
    private JButton previous, next;
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
        openOutput = preference("open", SegmentationOutput.ROIS);
        fileOutput = preference("files", SegmentationOutput.LABELS);
        previous = arrow("\u2039", -1);
        next = arrow("\u203a", 1);
        source.getSystemImagesRadio().addItemListener(e -> updateOutput());
        updateOutput();
    }

    public void setOpenImageRoisSupported(boolean supported) {
        openRois = supported;
        if (!supported) openOutput = SegmentationOutput.LABELS;
        updateOutput();
    }

    private SegmentationOutput preference(String source, SegmentationOutput fallback) {
        try { return SegmentationOutput.valueOf(preferences.get("segmentation." + source, fallback.name())); }
        catch (IllegalArgumentException e) { return fallback; }
    }

    private JButton arrow(String text, int direction) {
        JButton b = new JButton(text);
        YoloUiUtils.styleFlatSecondaryButton(b);
        b.setMargin(new Insets(0, 0, 0, 0));
        String tip = direction < 0 ? "Previous output" : "Next output";
        b.setToolTipText(tip);
        b.getAccessibleContext().setAccessibleName(tip);
        b.addActionListener(e -> {
            boolean files = segmentationSource.getSystemImagesRadio().isSelected();
            SegmentationOutput[] modes = SegmentationOutput.values();
            SegmentationOutput value = modes[Math.floorMod(output().ordinal() + direction, modes.length)];
            if (files) fileOutput = value; else openOutput = value;
            preferences.put("segmentation." + (files ? "files" : "open"), value.name());
            updateOutput();
        });
        add(b);
        return b;
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
        if (previous == null || next == null) return;
        boolean supported = openRois || segmentationSource.getSystemImagesRadio().isSelected()
                || output() == SegmentationOutput.LABELS;
        previous.setEnabled(!outputLocked);
        next.setEnabled(!outputLocked);
        runButton.setEnabled(sourceReady && supported && !outputLocked);
        String caption = "Run \u2192 " + output().title;
        YoloUiUtils.setFullText(runButton, caption);
        runButton.getAccessibleContext().setAccessibleName(caption);
        runButton.setToolTipText(supported ? caption : "Update the host adapter to add ROIs to an open image.");
        doLayout();
        repaint();
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
        boolean carousel = previous != null;
        int cancelW = Math.max(1, (w - GAP) / (carousel ? 3 : 2));
        int runW = Math.max(1, w - GAP - cancelW);
        cancelButton.setBounds(0, 0, cancelW, h);
        if (carousel) {
            int arrowW = Math.min(32, Math.max(1, runW / 7));
            previous.setBounds(cancelW + GAP, 0, arrowW, h);
            next.setBounds(w - arrowW, 0, arrowW, h);
            runW = Math.max(1, runW - 2 * arrowW);
            runButton.setBounds(cancelW + GAP + arrowW, 0, runW, h);
        } else runButton.setBounds(cancelW + GAP, 0, runW, h);
        YoloUiUtils.applyResponsiveText(cancelButton, cancelW - 8, h);
        YoloUiUtils.applyResponsiveText(runButton, runW - 8, h);
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
