package io.bioimage.modelrunner.model.python.envs;

import static org.junit.Assert.*;

import java.awt.Dimension;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.concurrent.atomic.AtomicInteger;

import javax.swing.JScrollPane;
import javax.swing.JTextPane;
import javax.swing.SwingUtilities;
import javax.swing.UIManager;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.gui.custom.yolo.TrainingLogPanel;
import io.bioimage.modelrunner.gui.custom.yolo.YoloHtmlLogPanel;

public class InstallationLogTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test public void followSurvivesAutomaticLayoutDuringEnvironmentSetup() throws Exception {
        org.junit.Assume.assumeFalse(java.awt.GraphicsEnvironment.isHeadless());
        for (boolean training : new boolean[] {false, true}) {
            javax.swing.JFrame[] window = new javax.swing.JFrame[1];
            javax.swing.JPanel[] holder = new javax.swing.JPanel[1];
            SwingUtilities.invokeAndWait(() -> {
                javax.swing.JFrame frame = window[0] = new javax.swing.JFrame();
                frame.setFocusableWindowState(false);
                frame.setLocation(-10000, -10000);
                holder[0] = training ? new TrainingLogPanel() : new YoloHtmlLogPanel();
                frame.add(holder[0]);
                frame.setSize(440, 240);
                frame.setVisible(true);
                if (training) ((TrainingLogPanel) holder[0]).setTrainingStatus(true, 0, 0, 2, 0L, Double.NaN);
                else ((YoloHtmlLogPanel) holder[0]).startRunTimer();
            });
            TrainingLogPanel train = training ? (TrainingLogPanel) holder[0] : null;
            YoloHtmlLogPanel inference = training ? null : (YoloHtmlLogPanel) holder[0];
            javax.swing.JEditorPane text = training ? (JTextPane) field(train, "textPane") : inference.getEditorPane();
            JScrollPane scroll = (JScrollPane) field(holder[0], "scrollPane");
            try (EnvironmentSetupLog.Scope scope = training ? train.openEnvironmentLog() : inference.openEnvironmentLog()) {
                EnvironmentSetupLog.Reporter reporter = new EnvironmentSetupLog.Reporter("test-env", null);
                reporter.announceSetup(false);
                for (int i = 0; i < 20; i++) {
                    if (training) train.appendLine("Preparing step " + i);
                    else inference.appendHtml("Preparing step " + i);
                    reporter.progress("Installing conda packages", i + 1, 20);
                    tick();
                    final int step = i;
                    SwingUtilities.invokeAndWait(() -> {
                        assertEquals("caret: training=" + training + ", step=" + step,
                                text.getDocument().getLength(), text.getCaretPosition());
                        assertTrue("scroll: training=" + training + ", step=" + step,
                                scroll.getVerticalScrollBar().getValue() + scroll.getVerticalScrollBar().getVisibleAmount()
                                        >= scroll.getVerticalScrollBar().getMaximum() - 4);
                    });
                }
                wheel(scroll, -3);
                tick();
                AtomicInteger readingPosition = new AtomicInteger();
                SwingUtilities.invokeAndWait(() -> {
                    readingPosition.set(scroll.getVerticalScrollBar().getValue());
                    assertTrue(readingPosition.get() + scroll.getVerticalScrollBar().getVisibleAmount()
                            < scroll.getVerticalScrollBar().getMaximum() - 4);
                });
                if (training) train.appendLine("A new message while the user reads earlier output");
                else inference.appendHtml("A new message while the user reads earlier output");
                tick();
                SwingUtilities.invokeAndWait(() -> assertEquals(readingPosition.get(), scroll.getVerticalScrollBar().getValue()));
                wheel(scroll, 1000);
                tick();
                SwingUtilities.invokeAndWait(() -> {
                    assertEquals(text.getDocument().getLength(), text.getCaretPosition());
                    assertTrue(scroll.getVerticalScrollBar().getValue() + scroll.getVerticalScrollBar().getVisibleAmount()
                            >= scroll.getVerticalScrollBar().getMaximum() - 4);
                });
                reporter.finish(true, "diagnostics");
            } finally {
                SwingUtilities.invokeAndWait(() -> {
                    if (training) train.setTrainingStatus(false, 0, 0, 2, 0L, Double.NaN);
                    else inference.stopRunTimer();
                    window[0].dispose();
                });
            }
        }
    }

    private static void wheel(JScrollPane scroll, int rotation) {
        java.awt.Toolkit.getDefaultToolkit().getSystemEventQueue().postEvent(new java.awt.event.MouseWheelEvent(
                scroll, java.awt.event.MouseEvent.MOUSE_WHEEL, System.currentTimeMillis(), 0, 10, 10, 0, false,
                java.awt.event.MouseWheelEvent.WHEEL_UNIT_SCROLL, 3, rotation));
    }

    @Test public void completedSetupClearsLiveLineWhileTheTaskContinues() throws Exception {
        for (boolean training : new boolean[] {false, true}) {
            for (boolean success : new boolean[] {false, true}) {
                javax.swing.JPanel[] holder = new javax.swing.JPanel[1];
                SwingUtilities.invokeAndWait(() -> {
                    if (training) {
                        TrainingLogPanel panel = new TrainingLogPanel();
                        panel.setTrainingStatus(true, 0, 0, 2, 0L, Double.NaN);
                        holder[0] = panel;
                    } else holder[0] = new YoloHtmlLogPanel();
                });
                TrainingLogPanel train = training ? (TrainingLogPanel) holder[0] : null;
                YoloHtmlLogPanel inference = training ? null : (YoloHtmlLogPanel) holder[0];
                javax.swing.JEditorPane text = training ? (JTextPane) field(train, "textPane") : inference.getEditorPane();
                try (EnvironmentSetupLog.Scope scope = training ? train.openEnvironmentLog() : inference.openEnvironmentLog()) {
                    EnvironmentSetupLog.Reporter reporter = new EnvironmentSetupLog.Reporter("test-env", null);
                    reporter.progress("Done", 1, 1);
                    tick();
                    SwingUtilities.invokeAndWait(() -> assertTrue(documentText(text).contains("Finalizing Python environment")));
                    reporter.finish(success, "diagnostics.log");
                    tick();
                    SwingUtilities.invokeAndWait(() -> {
                        if (training) train.appendLine("Continuing the task");
                        else inference.appendHtml("Continuing the task");
                        assertFalse(documentText(text), documentText(text).contains("Finalizing Python environment"));
                        assertTrue(documentText(text).contains("Continuing the task"));
                    });
                    // Buffered Appose output arriving after completion must not resurrect the spinner.
                    reporter.progress("Done", 1, 1);
                    reporter.output("INFO update: Installed environment 'default'");
                    tick();
                    SwingUtilities.invokeAndWait(() -> assertFalse(documentText(text).contains("Finalizing Python environment")));
                } finally {
                    SwingUtilities.invokeAndWait(() -> {
                        if (training) train.setTrainingStatus(false, 0, 0, 2, 0L, Double.NaN);
                        else inference.stopRunTimer();
                    });
                }
            }
        }
    }

    @Test public void bothLogsFillTheViewportWithWhiteForEmptyShortAndLongContent() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            Object previous = UIManager.get("Viewport.background");
            UIManager.put("Viewport.background", Color.MAGENTA);
            try {
                YoloHtmlLogPanel inference = new YoloHtmlLogPanel();
                TrainingLogPanel training = new TrainingLogPanel();
                for (javax.swing.JPanel panel : new javax.swing.JPanel[] {inference, training}) {
                    JScrollPane scroll = (JScrollPane) fieldUnchecked(panel, "scrollPane");
                    for (int width : new int[] {500, 220, 800}) {
                        panel.setSize(width, 240);
                        for (String message : new String[] {"", "Short message", "Long ".repeat(200), ""}) {
                            if (panel == inference) {
                                inference.clearLog();
                                if (!message.isEmpty()) inference.appendHtml(message);
                            } else {
                                training.clearLog();
                                if (!message.isEmpty()) training.appendLine(message);
                            }
                            for (int i = 0; i < 3; i++) {
                                panel.doLayout();
                                scroll.doLayout();
                                scroll.getViewport().doLayout();
                            }
                            javax.swing.JEditorPane text = (javax.swing.JEditorPane) scroll.getViewport().getView();
                            assertEquals(Color.WHITE, scroll.getViewport().getBackground());
                            assertTrue(text.isOpaque());
                            BufferedImage image = new BufferedImage(scroll.getViewport().getWidth(),
                                    scroll.getViewport().getHeight(), BufferedImage.TYPE_INT_RGB);
                            Graphics2D graphics = image.createGraphics();
                            scroll.getViewport().paint(graphics);
                            graphics.dispose();
                            assertEquals(Color.WHITE.getRGB(), image.getRGB(2, image.getHeight() - 3));
                            assertEquals(Color.WHITE.getRGB(), image.getRGB(image.getWidth() - 3, image.getHeight() - 3));
                        }
                    }
                }
            } finally {
                UIManager.put("Viewport.background", previous);
            }
        });
    }

    private static Object fieldUnchecked(Object target, String name) {
        try { return field(target, name); }
        catch (Exception e) { throw new AssertionError(e); }
    }

    private static String documentText(javax.swing.JEditorPane text) {
        try { return text.getDocument().getText(0, text.getDocument().getLength()); }
        catch (javax.swing.text.BadLocationException e) { throw new AssertionError(e); }
    }

    @Test public void trainingReplacesLiveLineAndKeepsOnlySummariesOnDisk() throws Exception {
        TrainingLogPanel[] holder = new TrainingLogPanel[1];
        SwingUtilities.invokeAndWait(() -> {
            holder[0] = new TrainingLogPanel();
            holder[0].setTrainingStatus(true, 0, 0, 2, 0L, Double.NaN);
        });
        TrainingLogPanel panel = holder[0];
        panel.startDiskLog(temporary.newFolder("model"));
        JTextPane text = (JTextPane) field(panel, "textPane");
        try (EnvironmentSetupLog.Scope scope = panel.openEnvironmentLog()) {
            EnvironmentSetupLog.Reporter reporter = new EnvironmentSetupLog.Reporter("test-env", null);
            reporter.progress("Downloading pixi", 1, 4);
            tick();
            SwingUtilities.invokeAndWait(() -> {
                assertTrue(text.getText(), text.getText().contains("25.0%"));
                assertFalse(text.getText().contains("Working"));
            });
            reporter.progress("Downloading pixi", 3, 4);
            tick();
            SwingUtilities.invokeAndWait(() -> {
                assertTrue(text.getText().contains("75.0%"));
                assertFalse(text.getText().contains("25.0%"));
            });
            String disk = new String(Files.readAllBytes(panel.getLogFile().toPath()), StandardCharsets.UTF_8);
            assertFalse(disk.contains("75.0%"));
            reporter.progress("Downloading pixi", 4, 4);
            tick();
            reporter.finish(true, "diagnostics.log");
        } finally {
            SwingUtilities.invokeAndWait(() -> panel.setTrainingStatus(false, 0, 0, 2, 0L, Double.NaN));
            panel.closeDiskLog();
        }
        String disk = new String(Files.readAllBytes(panel.getLogFile().toPath()), StandardCharsets.UTF_8);
        assertTrue(disk, disk.contains("100.0%"));
        assertTrue(disk.contains("environment ready"));
        assertFalse(disk.contains("75.0%"));
        assertFalse(text.getText().contains("Resolving dependencies /"));
    }

    @Test public void inferencePreservesScrollUpAndResumesFollowingAtBottom() throws Exception {
        YoloHtmlLogPanel[] holder = new YoloHtmlLogPanel[1];
        SwingUtilities.invokeAndWait(() -> {
            YoloHtmlLogPanel panel = holder[0] = new YoloHtmlLogPanel();
            panel.setSize(440, 240);
            panel.startRunTimer();
            for (int i = 0; i < 40; i++) panel.appendHtml("Previous log " + i);
        });
        YoloHtmlLogPanel panel = holder[0];
        JScrollPane scroll = (JScrollPane) field(panel, "scrollPane");
        settle(panel, scroll);
        SwingUtilities.invokeAndWait(() -> assertEquals(panel.getEditorPane().getDocument().getLength(),
                panel.getEditorPane().getCaretPosition()));
        SwingUtilities.invokeAndWait(() -> {
            panel.appendHtml("Queue a tail update before a user scroll");
            SwingUtilities.invokeLater(() -> scrollAsUser(scroll, 50));
        });
        settle(panel, scroll);
        SwingUtilities.invokeAndWait(() -> {
            assertEquals(50, scroll.getVerticalScrollBar().getValue());
            scrollAsUser(scroll, scroll.getVerticalScrollBar().getMaximum());
        });
        javax.swing.text.Document originalDocument = panel.getEditorPane().getDocument();
        AtomicInteger selection = new AtomicInteger();
        try (EnvironmentSetupLog.Scope scope = panel.openEnvironmentLog()) {
            EnvironmentSetupLog.Reporter reporter = new EnvironmentSetupLog.Reporter("test-env", null);
            reporter.progress("Installing conda packages", 1, 10);
            tick();
            settle(panel, scroll);
            SwingUtilities.invokeAndWait(() -> {
                assertEquals(0, scroll.getHorizontalScrollBar().getValue());
                assertTimestampSharesRow(panel.getEditorPane(), "Installing conda packages");
            });
            SwingUtilities.invokeAndWait(() -> {
                scrollAsUser(scroll, 50);
                selection.set(panel.getEditorPane().viewToModel2D(new java.awt.Point(80, 70)));
                panel.getEditorPane().select(selection.get(), selection.get() + 5);
            });
            reporter.progress("Installing conda packages", 7, 10);
            tick();
            panel.appendHtml("Another message");
            settle(panel, scroll);
            SwingUtilities.invokeAndWait(() -> {
                assertEquals(50, scroll.getVerticalScrollBar().getValue());
                assertEquals(selection.get(), panel.getEditorPane().getSelectionStart());
                assertEquals(selection.get() + 5, panel.getEditorPane().getSelectionEnd());
                assertSame(originalDocument, panel.getEditorPane().getDocument());
                String text = panel.getEditorPane().getText();
                assertTrue(text, text.contains("70.0%"));
                assertFalse(text.contains("10.0%"));
                scrollAsUser(scroll, scroll.getVerticalScrollBar().getMaximum());
            });
            // Reaching the bottom must restore the caret even without another backend message.
            settle(panel, scroll);
            SwingUtilities.invokeAndWait(() -> assertEquals(panel.getEditorPane().getDocument().getLength(),
                    panel.getEditorPane().getCaretPosition()));
            reporter.progress("Installing conda packages", 9, 10);
            tick();
            settle(panel, scroll);
            SwingUtilities.invokeAndWait(() -> {
                assertTrue(scroll.getVerticalScrollBar().getValue() + scroll.getVerticalScrollBar().getVisibleAmount()
                        >= scroll.getVerticalScrollBar().getMaximum() - 4);
                assertEquals(panel.getEditorPane().getDocument().getLength(), panel.getEditorPane().getCaretPosition());
                assertEquals(0, scroll.getHorizontalScrollBar().getValue());
            });
            reporter.finish(false, "/path/to/diagnostics.log");
            tick();
            assertTrue(panel.getEditorPane().getText().contains("diagnostics.log"));
        } finally {
            SwingUtilities.invokeAndWait(panel::stopRunTimer);
        }
    }

    @Test public void inferenceRowsDoNotWrapOrRebuildTheDocumentWhenBounded() throws Exception {
        YoloHtmlLogPanel[] holder = new YoloHtmlLogPanel[1];
        SwingUtilities.invokeAndWait(() -> {
            holder[0] = new YoloHtmlLogPanel();
            holder[0].setSize(180, 160);
            holder[0].appendHtml("Starting inference on 1 image(s).");
            holder[0].appendHtml("Loading model: /a/very/long/path/to/a/model/checkpoint.pt");
        });
        YoloHtmlLogPanel panel = holder[0];
        JScrollPane scroll = (JScrollPane) field(panel, "scrollPane");
        settle(panel, scroll);
        javax.swing.text.Document document = panel.getEditorPane().getDocument();
        SwingUtilities.invokeAndWait(() -> {
            try {
                String text = document.getText(0, document.getLength());
                int message = text.indexOf("Loading model:");
                int timestamp = text.lastIndexOf('[', message);
                assertEquals(panel.getEditorPane().modelToView2D(timestamp).getY(),
                        panel.getEditorPane().modelToView2D(message).getY(), 1.0);
                assertEquals(panel.getEditorPane().modelToView2D(timestamp).getY(),
                        panel.getEditorPane().modelToView2D(text.indexOf("checkpoint.pt")).getY(), 1.0);
                assertTrue("Long rows scroll horizontally, not wrap timestamps onto another line",
                        panel.getEditorPane().getWidth() > scroll.getViewport().getWidth());
                for (int i = 0; i < 300; i++) panel.appendHtml(String.format("Entry%03d::", i));
                assertSame(document, panel.getEditorPane().getDocument());
                text = document.getText(0, document.getLength());
                assertFalse(text.contains("Entry049::"));
                assertTrue(text.contains("Entry050::"));
                assertTrue(text.contains("Entry299::"));
                assertEquals(250, text.split("Entry", -1).length - 1);
            } catch (javax.swing.text.BadLocationException e) {
                throw new AssertionError(e);
            }
        });
    }

    @Test public void trainingLiveProgressPreservesSelectionAndRestoresCaretAtBottom() throws Exception {
        TrainingLogPanel[] holder = new TrainingLogPanel[1];
        SwingUtilities.invokeAndWait(() -> {
            TrainingLogPanel panel = holder[0] = new TrainingLogPanel();
            panel.setSize(240, 160);
            for (int i = 0; i < 40; i++) panel.appendLine("Existing training entry " + i);
            panel.setTrainingStatus(true, 0, 0, 2, 0L, Double.NaN);
        });
        TrainingLogPanel panel = holder[0];
        JScrollPane scroll = (JScrollPane) field(panel, "scrollPane");
        JTextPane text = (JTextPane) field(panel, "textPane");
        AtomicInteger selection = new AtomicInteger();
        try (EnvironmentSetupLog.Scope scope = panel.openEnvironmentLog()) {
            EnvironmentSetupLog.Reporter reporter = new EnvironmentSetupLog.Reporter("test-env", null);
            reporter.progress("Installing conda packages", 1, 10);
            tick();
            settle(panel, scroll, text);
            SwingUtilities.invokeAndWait(() -> {
                panel.appendLine("Queue a tail update before a user scroll");
                SwingUtilities.invokeLater(() -> scrollAsUser(scroll, 50));
            });
            settle(panel, scroll, text);
            SwingUtilities.invokeAndWait(() -> {
                assertEquals(50, scroll.getVerticalScrollBar().getValue());
                scrollAsUser(scroll, scroll.getVerticalScrollBar().getMaximum());
            });
            settle(panel, scroll, text);
            SwingUtilities.invokeAndWait(() -> {
                assertEquals(text.getDocument().getLength(), text.getCaretPosition());
                assertEquals(0, scroll.getHorizontalScrollBar().getValue());
                assertTimestampSharesRow(text, "Installing conda packages");
                scrollAsUser(scroll, 50);
                selection.set(text.viewToModel2D(new java.awt.Point(80, 70)));
                text.select(selection.get(), selection.get() + 5);
            });
            reporter.progress("Installing conda packages", 7, 10);
            tick();
            panel.appendLine("Still installing");
            settle(panel, scroll, text);
            SwingUtilities.invokeAndWait(() -> {
                assertEquals(50, scroll.getVerticalScrollBar().getValue());
                assertEquals(selection.get(), text.getSelectionStart());
                assertEquals(selection.get() + 5, text.getSelectionEnd());
                scrollAsUser(scroll, scroll.getVerticalScrollBar().getMaximum());
            });
            settle(panel, scroll, text);
            SwingUtilities.invokeAndWait(() -> assertEquals(text.getDocument().getLength(), text.getCaretPosition()));
            reporter.finish(true, "diagnostics");
        } finally {
            SwingUtilities.invokeAndWait(() -> panel.setTrainingStatus(false, 0, 0, 2, 0L, Double.NaN));
        }
    }

    private static void scrollAsUser(JScrollPane scroll, int value) {
        javax.swing.JScrollBar bar = scroll.getVerticalScrollBar();
        bar.setValueIsAdjusting(true);
        bar.setValue(value);
        bar.setValueIsAdjusting(false);
    }

    private static void assertTimestampSharesRow(javax.swing.JEditorPane text, String message) {
        try {
            String content = text.getDocument().getText(0, text.getDocument().getLength());
            int start = content.indexOf(message);
            int timestamp = content.lastIndexOf('[', start);
            assertTrue(start >= 0 && timestamp >= 0);
            assertEquals(text.modelToView2D(timestamp).getY(), text.modelToView2D(start).getY(), 2.0);
        } catch (javax.swing.text.BadLocationException e) {
            throw new AssertionError(e);
        }
    }

    private static void settle(TrainingLogPanel panel, JScrollPane scroll, JTextPane text) throws Exception {
        for (int i = 0; i < 4; i++) SwingUtilities.invokeAndWait(() -> {
            panel.doLayout();
            scroll.doLayout();
            Dimension extent = scroll.getViewport().getExtentSize(), preferred = text.getPreferredSize();
            text.setSize(Math.max(extent.width, preferred.width), Math.max(extent.height, preferred.height));
            scroll.getViewport().doLayout();
            scroll.doLayout();
        });
    }

    private static void tick() throws Exception {
        Thread.sleep(240);
        SwingUtilities.invokeAndWait(() -> {});
    }

    private static void settle(YoloHtmlLogPanel panel, JScrollPane scroll) throws Exception {
        for (int i = 0; i < 4; i++) SwingUtilities.invokeAndWait(() -> {
            panel.doLayout();
            scroll.doLayout();
            Dimension extent = scroll.getViewport().getExtentSize();
            Dimension preferred = panel.getEditorPane().getPreferredSize();
            panel.getEditorPane().setSize(Math.max(extent.width, preferred.width), Math.max(extent.height, preferred.height));
            scroll.getViewport().doLayout();
            scroll.doLayout();
        });
    }

    private static Object field(Object target, String name) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        return field.get(target);
    }
}
