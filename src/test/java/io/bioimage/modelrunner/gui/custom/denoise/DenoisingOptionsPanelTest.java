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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.awt.Component;
import java.util.concurrent.atomic.AtomicInteger;

import javax.swing.JCheckBox;
import javax.swing.JRadioButton;
import javax.swing.SwingUtilities;

import org.junit.Test;

public class DenoisingOptionsPanelTest {
    @Test
    public void showsBalancedHighOnlyForZsN2NAndResetsWhenSwitchingMethod() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            DenoisingOptionsPanel panel = new DenoisingOptionsPanel(new JCheckBox());
            JRadioButton high = button(panel, "Balanced-high");
            assertEquals(DenoisingEffort.BALANCED, panel.getEffort());
            assertFalse(high.isVisible());
            assertFalse(high.isEnabled());

            for (DenoisingMethod other : new DenoisingMethod[] {
                    DenoisingMethod.FAST, DenoisingMethod.CONSERVATIVE, DenoisingMethod.CORRELATED}) {
                panel.getMethodComboBox().setSelectedItem(DenoisingMethod.ADAPTIVE);
                assertTrue(high.isVisible());
                assertTrue(high.isEnabled());
                high.doClick();
                assertEquals(DenoisingEffort.BALANCED_HIGH, panel.getEffort());
                panel.getMethodComboBox().setSelectedItem(other);
                assertFalse(high.isVisible());
                assertFalse(high.isEnabled());
                assertEquals(DenoisingEffort.BALANCED, panel.getEffort());
            }
        });
    }

    @Test
    public void notifiesEffortChangesAndDisablesAllChoicesDuringWork() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            DenoisingOptionsPanel panel = new DenoisingOptionsPanel(new JCheckBox());
            panel.getMethodComboBox().setSelectedItem(DenoisingMethod.ADAPTIVE);
            AtomicInteger changes = new AtomicInteger();
            panel.addEffortActionListener(e -> changes.incrementAndGet());
            button(panel, "Balanced-high").doClick();
            assertEquals(1, changes.get());
            panel.setInteractionEnabled(false);
            for (Component component : panel.getEffortRow().getComponents()) {
                if (component instanceof JRadioButton) assertFalse(component.isEnabled());
            }
            button(panel, "Quick").doClick();
            assertEquals(1, changes.get());
            panel.setInteractionEnabled(true);
            button(panel, "Quick").doClick();
            assertEquals(2, changes.get());
            assertEquals(DenoisingEffort.QUICK, panel.getEffort());
        });
    }

    @Test
    public void fitsThreeOrFourOptionsInTheSameRowAtDifferentWidths() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            DenoisingOptionsPanel panel = new DenoisingOptionsPanel(new JCheckBox());
            for (DenoisingMethod method : DenoisingMethod.values()) {
                panel.getMethodComboBox().setSelectedItem(method);
                for (int width : new int[] {360, 600, 900}) {
                    panel.getEffortRow().setSize(width, 32);
                    panel.getEffortRow().doLayout();
                    int end = 0;
                    int count = 0;
                    for (Component component : panel.getEffortRow().getComponents()) {
                        if (!component.isVisible()) continue;
                        assertTrue(component.getX() >= end);
                        assertTrue(component.getWidth() > 0);
                        assertEquals(32, component.getHeight());
                        end = component.getX() + component.getWidth();
                        assertTrue(end <= width);
                        if (component instanceof JRadioButton) count++;
                    }
                    assertEquals(method == DenoisingMethod.ADAPTIVE ? 4 : 3, count);
                }
            }
        });
    }

    private static JRadioButton button(DenoisingOptionsPanel panel, String text) {
        for (Component component : panel.getEffortRow().getComponents()) {
            if (component instanceof JRadioButton && text.equals(((JRadioButton) component).getText()))
                return (JRadioButton) component;
        }
        throw new AssertionError("Missing effort: " + text);
    }
}
