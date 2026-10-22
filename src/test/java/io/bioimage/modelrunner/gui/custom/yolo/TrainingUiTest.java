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
package io.bioimage.modelrunner.gui.custom.yolo;

import static org.junit.Assert.*;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.concurrent.atomic.AtomicInteger;

import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JTextPane;
import javax.swing.SwingUtilities;

import org.junit.Test;

import io.bioimage.modelrunner.gui.custom.stardist.StardistValidationPreviewPanel;
import io.bioimage.modelrunner.gui.custom.unet.UnetTrainPanel;
import net.imglib2.img.array.ArrayImgs;

public class TrainingUiTest {
    @Test
    public void trainingViewButtonsHighlightOnlyTheVisibleView() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            for (BaseTrainPanel panel : new BaseTrainPanel[] {new YoloTrainPanel(), new StardistTrainPanel(),
                    new UnetTrainPanel() { private static final long serialVersionUID = 1L; }}) {
                assertSelectedView(panel, panel.lossButton);
                for (JButton button : new JButton[] {panel.metricButton, panel.validationPreviewButton,
                        panel.logButton, panel.lossButton}) {
                    button.doClick();
                    assertSelectedView(panel, button);
                    button.doClick();
                    assertSelectedView(panel, button);
                }
                panel.setTrainingRunning(true);
                assertSelectedView(panel, panel.logButton);
                panel.validationPreviewButton.doClick();
                panel.setTrainingRunning(true);
                assertSelectedView(panel, panel.validationPreviewButton);
                panel.setFullValidationAvailable(true);
                panel.setFullValidationState(true, true);
                assertSelectedView(panel, panel.validationPreviewButton);
                panel.setTrainingRunning(false);
                assertSelectedView(panel, panel.logButton);
            }
        });
    }

    private static void assertSelectedView(BaseTrainPanel panel, JButton selected) {
        JButton[] buttons = {panel.lossButton, panel.metricButton, panel.validationPreviewButton, panel.logButton};
        JPanel[] views = {panel.lossGraphPanel, panel.metricGraphPanel, panel.validationPreviewPanel, panel.trainingLogPanel};
        for (int i = 0; i < buttons.length; i++) {
            boolean active = buttons[i] == selected;
            assertEquals(buttons[i].getText(), active, buttons[i].isSelected());
            assertEquals(buttons[i].getText(), active, views[i].isVisible());
            assertEquals(active ? YoloUiUtils.TOGGLE_ACTIVE_BUTTON_BG : YoloUiUtils.SECONDARY_BUTTON_BG,
                    buttons[i].getBackground());
            assertEquals(active ? YoloUiUtils.PRIMARY_BUTTON_FG : YoloUiUtils.SECONDARY_BUTTON_FG,
                    buttons[i].getForeground());
        }
    }

    @Test
    public void startingTrainingRestoresLogCaretAndFollowing() throws Exception {
        BaseTrainPanel[] panels = new BaseTrainPanel[3];
        SwingUtilities.invokeAndWait(() -> {
            panels[0] = new YoloTrainPanel();
            panels[1] = new StardistTrainPanel();
            panels[2] = new UnetTrainPanel() { private static final long serialVersionUID = 1L; };
        });
        for (BaseTrainPanel panel : panels) {
            TrainingLogPanel log = panel.trainingLogPanel;
            JScrollPane scroll = (JScrollPane) field(log, "scrollPane");
            JTextPane text = (JTextPane) field(log, "textPane");
            SwingUtilities.invokeAndWait(() -> {
                log.setSize(440, 240);
                for (int i = 0; i < 100; i++) log.appendLine("Previous log " + i);
            });
            settle(log, scroll, text);
            SwingUtilities.invokeAndWait(() -> {
                scroll.getVerticalScrollBar().setValue(50);
                text.setCaretPosition(0);
                panel.lossButton.doClick();
                panel.setTrainingRunning(true);
                assertTrue(log.isVisible());
                log.appendLine("Starting training");
            });
            settle(log, scroll, text);
            SwingUtilities.invokeAndWait(() -> {
                assertBottom(scroll);
                assertEquals(text.getDocument().getLength(), text.getCaretPosition());
            });
            // Backend callbacks can append off the event dispatch thread.
            log.appendLine("Next training step");
            settle(log, scroll, text);
            SwingUtilities.invokeAndWait(() -> {
                assertBottom(scroll);
                assertEquals(text.getDocument().getLength(), text.getCaretPosition());
            });

            AtomicInteger position = new AtomicInteger();
            SwingUtilities.invokeAndWait(() -> {
                scroll.getVerticalScrollBar().setValue(50);
                position.set(scroll.getVerticalScrollBar().getValue());
                panel.setTrainingRunning(true);
                log.appendLine("Reading earlier messages during training");
            });
            settle(log, scroll, text);
            SwingUtilities.invokeAndWait(() -> {
                assertEquals(position.get(), scroll.getVerticalScrollBar().getValue());
                panel.setTrainingRunning(false);
                panel.metricButton.doClick();
                panel.setTrainingRunning(true);
                assertTrue(log.isVisible());
                log.appendLine("Another training run");
            });
            settle(log, scroll, text);
            SwingUtilities.invokeAndWait(() -> {
                assertBottom(scroll);
                assertEquals(text.getDocument().getLength(), text.getCaretPosition());
                panel.setTrainingRunning(false);
            });
        }
    }

    @Test
    public void logLifecycleAndFollowingRespectUserNavigation() throws Exception {
        TrainingLogPanel[] holder = new TrainingLogPanel[1];
        SwingUtilities.invokeAndWait(() -> {
            for (BaseTrainPanel panel : new BaseTrainPanel[] {new YoloTrainPanel(), new StardistTrainPanel(),
                    new UnetTrainPanel() { private static final long serialVersionUID = 1L; }}) {
                panel.setTrainingRunning(true);
                assertTrue(panel.trainingLogPanel.isVisible());
                panel.lossButton.doClick();
                panel.setTrainingRunning(true);
                assertTrue("Status updates must not steal the selected view", panel.lossGraphPanel.isVisible());
                panel.setTrainingRunning(false);
                assertTrue(panel.trainingLogPanel.isVisible());
            }
            holder[0] = new TrainingLogPanel();
            holder[0].setSize(440, 240);
            for (int i = 0; i < 100; i++) holder[0].appendLine("Line " + i);
        });
        TrainingLogPanel log = holder[0];
        JScrollPane scroll = (JScrollPane) field(log, "scrollPane");
        JTextPane text = (JTextPane) field(log, "textPane");
        settle(log, scroll, text);
        SwingUtilities.invokeAndWait(() -> assertBottom(scroll));

        AtomicInteger position = new AtomicInteger();
        AtomicInteger selection = new AtomicInteger();
        SwingUtilities.invokeAndWait(() -> {
            // Scroll up after a tail update is queued but before it executes.
            log.appendLine("Queued update");
            scroll.getVerticalScrollBar().setValue(150);
            position.set(scroll.getVerticalScrollBar().getValue());
            selection.set(text.viewToModel2D(new java.awt.Point(12, position.get() + 20)));
            text.select(selection.get(), selection.get() + 5);
            log.appendLine("Do not move the reader or their selection");
        });
        settle(log, scroll, text);
        SwingUtilities.invokeAndWait(() -> {
            assertEquals(position.get(), scroll.getVerticalScrollBar().getValue());
            assertEquals(selection.get(), text.getSelectionStart());
            assertEquals(selection.get() + 5, text.getSelectionEnd());
            log.setTrainingStatus(true, 1, 100, 2, 1000, 1);
            log.setTrainingStatus(false, 1, 100, 2, 1000, 1);
            log.appendLine("Activity updates must not pull the reader down either");
        });
        settle(log, scroll, text);
        SwingUtilities.invokeAndWait(() -> {
            assertEquals(position.get(), scroll.getVerticalScrollBar().getValue());
            scroll.getVerticalScrollBar().setValue(scroll.getVerticalScrollBar().getMaximum());
            log.appendLine("Follow again");
        });
        settle(log, scroll, text);
        SwingUtilities.invokeAndWait(() -> {
            assertBottom(scroll);
            log.setSize(300, 160);
        });
        settle(log, scroll, text);
        SwingUtilities.invokeAndWait(() -> {
            assertBottom(scroll);
            scroll.getVerticalScrollBar().setValue(50);
            log.clearLog();
            for (int i = 0; i < 100; i++) log.appendLine("New run " + i);
        });
        settle(log, scroll, text);
        SwingUtilities.invokeAndWait(() -> {
            assertBottom(scroll);
            for (int i = 100; i < 2000; i++) log.appendLine("New run " + i);
        });
        settle(log, scroll, text);
        SwingUtilities.invokeAndWait(() -> {
            scroll.getVerticalScrollBar().setValue(500);
            position.set(scroll.getVerticalScrollBar().getValue());
            log.appendLine("Trim the oldest line while keeping the reader's place");
        });
        settle(log, scroll, text);
        SwingUtilities.invokeAndWait(() -> {
            int value = scroll.getVerticalScrollBar().getValue();
            assertTrue("Compensate for the removed first line", value < position.get() && value > position.get() - 30);
        });
    }

    @Test
    public void annotationColorsAreIndependentOfReadableTitles() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            try {
                YoloValidationPreviewPanel boxes = new YoloValidationPreviewPanel();
                invoke(boxes, "setSelectedBoxColor", Color.class, Color.CYAN);
                boxes.clearPreview();
                JButton picker = (JButton) field(boxes, "annotationColorButton");
                assertEquals("Keep the preference for the next run", Color.CYAN, picker.getForeground());
                boxes.setSize(600, 450);
                boxes.doLayout();
                int large = picker.getWidth();
                boxes.setSize(220, 180);
                boxes.doLayout();
                assertTrue(picker.getWidth() < large);

                StardistValidationPreviewPanel dense = new StardistValidationPreviewPanel();
                field(dense, "currentImage", ArrayImgs.unsignedBytes(16, 16));
                byte[] mask = new byte[256];
                mask[8 * 16 + 8] = 1;
                field(dense, "currentPrediction", ArrayImgs.unsignedBytes(mask, 16, 16));
                invoke(dense, "setAnnotationColor", Color.class, Color.CYAN);
                BufferedImage overlay = (BufferedImage) invoke(dense, "buildOverlay", null, null);
                Color pixel = new Color(overlay.getRGB(8, 8));
                assertEquals(0, pixel.getRed());
                assertTrue(pixel.getGreen() > 150 && pixel.getBlue() > 150);
                dense.clearPreview();
                assertEquals(Color.CYAN, field(dense, "predictionColor"));

                for (Object panel : new Object[] {boxes, dense}) {
                    YoloImageDisplayPanel image = (YoloImageDisplayPanel) field(panel, "imagePanel");
                    image.setSize(300, 180);
                    image.setBufferedImage(new BufferedImage(300, 180, BufferedImage.TYPE_INT_RGB), "Sample", false);
                    image.setTitleColor(Color.RED);
                    BufferedImage rendered = new BufferedImage(300, 180, BufferedImage.TYPE_INT_RGB);
                    Graphics2D graphics = rendered.createGraphics();
                    image.paint(graphics);
                    graphics.dispose();
                    Rectangle area = (Rectangle) field(image, "imageDrawArea");
                    boolean whiteText = false;
                    for (int y = area.y + 4; y < area.y + 20; y++) {
                        for (int x = area.x + 6; x < area.x + 45; x++) {
                            if ((rendered.getRGB(x, y) & 0xffffff) == 0xffffff) whiteText = true;
                        }
                    }
                    assertTrue("Annotation/title color setters cannot change the white label", whiteText);
                    Color backdrop = new Color(rendered.getRGB(area.x + 3, area.y + 13));
                    assertTrue("The label needs a dark backing", backdrop.getRed() < 120);
                    image.clearImage();
                }
            } catch (Exception e) {
                throw new AssertionError(e);
            }
        });
    }

    private static void settle(TrainingLogPanel log, JScrollPane scroll, JTextPane text) throws Exception {
        for (int i = 0; i < 4; i++) SwingUtilities.invokeAndWait(() -> {
            log.doLayout();
            scroll.doLayout();
            Dimension extent = scroll.getViewport().getExtentSize();
            Dimension preferred = text.getPreferredSize();
            text.setSize(Math.max(extent.width, preferred.width), Math.max(extent.height, preferred.height));
            scroll.getViewport().doLayout();
            scroll.doLayout();
        });
    }

    private static void assertBottom(JScrollPane scroll) {
        JScrollBar bar = scroll.getVerticalScrollBar();
        assertEquals(bar.getMaximum() - bar.getVisibleAmount(), bar.getValue());
    }

    private static Object field(Object target, String name) throws Exception {
        Class<?> type = target instanceof YoloImageDisplayPanel ? YoloImageDisplayPanel.class : target.getClass();
        Field field = type.getDeclaredField(name);
        field.setAccessible(true);
        return field.get(target);
    }

    private static void field(Object target, String name, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static Object invoke(Object target, String name, Class<?> parameter, Object value) throws Exception {
        Method method = parameter == null ? target.getClass().getDeclaredMethod(name)
                : target.getClass().getDeclaredMethod(name, parameter);
        method.setAccessible(true);
        return parameter == null ? method.invoke(target) : method.invoke(target, value);
    }
}
