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
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import io.bioimage.modelrunner.gui.custom.yolo.YoloImageDisplayPanel;

public class DenoisingViewportTest {
    @Test
    public void appliesSharedViewportState() {
        YoloImageDisplayPanel panel = new YoloImageDisplayPanel();
        panel.setViewport(new YoloImageDisplayPanel.Viewport(2.0d, 12.0d, -4.0d, true));
        assertEquals(2.0d, panel.getViewport().getZoom(), 0.0d);
        assertEquals(12.0d, panel.getViewport().getPanX(), 0.0d);
        assertEquals(-4.0d, panel.getViewport().getPanY(), 0.0d);
        assertTrue(panel.getViewport().isExpandedToFill());
    }
}
