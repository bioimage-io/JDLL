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
package io.bioimage.modelrunner.gui.custom;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;

import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.custom.crossgoose.CrossGooseBackend;

/** Cross-GOOSE instance-segmentation interface. */
public final class CrossGoosePluginUI extends UNetPluginUI {

    private static final long serialVersionUID = 1L;

    public CrossGoosePluginUI(ConsumerInterface consumer, GuiAdapter adapter) {
        super(consumer, adapter, new CrossGooseBackend());
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Cross-GOOSE Plugin");
            frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            frame.getContentPane().add(new CrossGoosePluginUI(null, null));
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
            frame.setResizable(true);
            frame.setSize(550, 700);
        });
    }
}
