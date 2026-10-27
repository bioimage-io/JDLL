/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * #L%
 */
package io.bioimage.modelrunner.gui.custom;

import static org.junit.Assert.*;

import java.awt.geom.Rectangle2D;
import java.io.File;

import org.junit.Test;

import io.bioimage.modelrunner.gui.custom.cellpose.CellposeInferenceService;

public class SizeHintsTest {
    @Test public void sharedHintsAndIndependentExceptionsNeverDependOnZ() {
        SizeHints hints = new SizeHints();
        File a = new File("dataset/a.tif"), b = new File("dataset/b.tif");
        Rectangle2D.Double shared = new Rectangle2D.Double(12, 15, 30, 40);
        Rectangle2D.Double local = new Rectangle2D.Double(25, 30, 60, 80);
        assertNull(hints.get(a, 0));
        hints.set(a, 0, shared);
        assertEquals(shared, hints.get(b, 7));
        hints.useShared(b, 1, false);
        assertEquals(shared, hints.get(b, 1));
        hints.set(b, 1, local);
        hints.set(a, 0, new Rectangle2D.Double(1, 2, 10, 20));
        assertEquals(local, hints.get(b, 1));
        assertEquals(10, hints.get(a, 0).width, 0);
        assertEquals(10, hints.get(b, 2).width, 0);
        hints.useShared(b, 1, true);
        assertEquals(10, hints.get(b, 1).width, 0);
        hints.useShared(b, 1, false);
        hints.set(b, 1, local);
        hints.clear(a, 0);
        assertNull(hints.get(a, 0));
        assertEquals(local, hints.get(b, 1));
        hints.clear(b, 1);
        assertNull(hints.get(b, 1));
        assertTrue(hints.usesShared(b, 1));
        hints.useShared(a, 1, false);
        hints.set(b, 0, shared);
        assertEquals("An undrawn exception still falls back to the shared size", shared, hints.get(a, 1));
    }

    @Test public void runSnapshotsAreIndependentAndCellposeResolvesDiameterPerTimePoint() {
        SizeHints hints = new SizeHints();
        File a = new File("dataset/a.tif"), b = new File("dataset/b.tif");
        Rectangle2D.Double rectangle = new Rectangle2D.Double(4, 7, 30, 30);
        hints.set(a, 0, rectangle);
        rectangle.width = 99;
        hints.useShared(b, 1, false);
        hints.set(b, 1, new Rectangle2D.Double(0, 0, 60, 60));
        SizeHints.Snapshot snapshot = hints.snapshot();
        hints.reset();
        assertEquals(30, snapshot.forImage(a).apply(2).width, 0);
        snapshot.forImage(a).apply(2).width = 777;
        assertEquals(30, snapshot.forImage(a).apply(2).width, 0);
        CellposeInferenceService.Run run = new CellposeInferenceService.Run(
                "cyto3", new int[] {0, 0}, null, "cpu", false).withSizeHints(snapshot.forImage(b));
        assertEquals(30, run.diameterAt(0), 0);
        assertEquals(60, run.diameterAt(1), 0);
        assertEquals(30, run.diameterAt(2), 0);
        assertEquals(30, snapshot.forImage(new File("dataset/../dataset/a.tif")).apply(0).width, 0);
    }
}
