/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * #L%
 */
package io.bioimage.modelrunner.gui.custom.denoise;

import javax.swing.JButton;
import javax.swing.JPanel;

import io.bioimage.modelrunner.gui.custom.yolo.YoloUiUtils;

public final class DenoisingActionPanel extends JPanel {
    private static final long serialVersionUID = 1L;
    private static final int GAP = 6;
    private final JButton cancel = new JButton("Cancel");
    private final JButton preview = new JButton("Preview");
    private final JButton run = new JButton("Run");

    public DenoisingActionPanel() {
        setLayout(null);
        setOpaque(false);
        YoloUiUtils.styleFlatSecondaryButton(cancel);
        YoloUiUtils.styleFlatSecondaryButton(preview);
        YoloUiUtils.styleFlatPrimaryButton(run);
        cancel.setEnabled(false);
        add(cancel); add(preview); add(run);
    }

    @Override public void doLayout() {
        int w = Math.max(1, (getWidth() - 2 * GAP) / 3);
        cancel.setBounds(0, 0, w, getHeight());
        preview.setBounds(w + GAP, 0, w, getHeight());
        run.setBounds(2 * (w + GAP), 0, Math.max(1, getWidth() - 2 * (w + GAP)), getHeight());
        YoloUiUtils.applyResponsiveText(cancel, w - 8, getHeight());
        YoloUiUtils.applyResponsiveText(preview, w - 8, getHeight());
        YoloUiUtils.applyResponsiveText(run, run.getWidth() - 8, getHeight());
    }

    public JButton getCancelButton() { return cancel; }
    public JButton getPreviewButton() { return preview; }
    public JButton getRunButton() { return run; }
}
