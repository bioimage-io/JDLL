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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.awt.Component;
import java.awt.Container;

import org.junit.Test;

public class DenoisingLayoutTest {

    @Test
    public void placesAccelerationInNoiseRowAndLogAboveActions() {
        DenoisingGUI gui = new DenoisingGUI(null);
        gui.setSize(900, 720);
        layoutTree(gui);

        assertSame(gui.getOptionsPanel().getNoiseRow(), gui.getAccelerationCheckBox().getParent());
        assertEquals(gui.getOptionsPanel().getMethodRow().getHeight() * 2,
                gui.getImageSourcePanel().getHeight());
        assertTrue(gui.getStatusPanel().getHeight()
                < gui.getOptionsPanel().getMethodRow().getHeight());
        assertTrue(gui.getStatusPanel().getY()
                > gui.getLogPanel().getY() + gui.getLogPanel().getHeight());
        assertTrue(gui.getActionPanel().getY()
                > gui.getStatusPanel().getY() + gui.getStatusPanel().getHeight());
    }

    @Test
    public void keepsProgressVisibleAndUsesTheWholeRow() {
        DenoisingStatusPanel panel = new DenoisingStatusPanel();
        panel.setSize(600, 22);
        panel.doLayout();
        panel.setProgress(3, 10);

        assertTrue(panel.getProgressBar().isVisible());
        assertEquals(600, panel.getProgressBar().getWidth());
        assertEquals(3, panel.getProgressBar().getValue());
        assertEquals(10, panel.getProgressBar().getMaximum());
    }

    private static void layoutTree(Container container) {
        container.doLayout();
        for (Component child : container.getComponents()) {
            if (child instanceof Container) layoutTree((Container) child);
        }
    }
}
