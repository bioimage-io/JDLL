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

import javax.swing.JPanel;
import javax.swing.JProgressBar;

public final class DenoisingStatusPanel extends JPanel {
    private static final long serialVersionUID = 1L;
    private final JProgressBar progress = new JProgressBar();

    public DenoisingStatusPanel() {
        setLayout(null);
        setOpaque(false);
        progress.setStringPainted(true);
        progress.setMinimum(0);
        progress.setMaximum(100);
        progress.setValue(0);
        add(progress);
    }

    public void setProgress(long current, long maximum) {
        if (maximum > 0L) {
            progress.setIndeterminate(false);
            progress.setMaximum((int) Math.min(Integer.MAX_VALUE, maximum));
            progress.setValue((int) Math.min(progress.getMaximum(), Math.max(0L, current)));
            progress.setString(null);
        } else {
            progress.setIndeterminate(true);
            progress.setString("");
        }
    }

    public void setIdle() {
        progress.setIndeterminate(false);
        progress.setMaximum(100);
        progress.setValue(0);
        progress.setString(null);
    }

    public void setComplete() {
        progress.setIndeterminate(false);
        progress.setMaximum(100);
        progress.setValue(100);
        progress.setString(null);
    }

    public JProgressBar getProgressBar() {
        return progress;
    }

    @Override public void doLayout() {
        int h = Math.max(1, getHeight());
        progress.setBounds(0, 1, Math.max(1, getWidth()), Math.max(1, h - 2));
    }
}
