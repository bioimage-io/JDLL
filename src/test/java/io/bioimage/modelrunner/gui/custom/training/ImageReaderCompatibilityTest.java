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
package io.bioimage.modelrunner.gui.custom.training;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.fail;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

import javax.imageio.ImageIO;
import javax.swing.JComponent;
import javax.swing.JButton;
import javax.swing.JScrollBar;
import javax.swing.SwingUtilities;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.gui.custom.CellposePluginUI;
import io.bioimage.modelrunner.gui.custom.ConsumerInterface;
import io.bioimage.modelrunner.gui.custom.PreviewReader;
import io.bioimage.modelrunner.gui.custom.CrossGoosePluginUI;
import io.bioimage.modelrunner.gui.custom.StarDistPluginUI;
import io.bioimage.modelrunner.gui.custom.UNetPluginUI;
import io.bioimage.modelrunner.gui.custom.YOLOPluginUI;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageDisplayPanel;
import io.bioimage.modelrunner.gui.custom.yolo.YoloInferencePanel;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageSourcePanel;
import io.bioimage.modelrunner.model.detection.Detection;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/** Test the actual disk-entry readers without constructing a GUI or loading a model. */
public class ImageReaderCompatibilityTest {
    @Rule public TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void sharedAndIndividualSizeHintsFollowFilesAndTimeButNeverZ() throws Exception {
        File first = folder.newFile("first.tif"), second = folder.newFile("second.tif");
        ImageConsumer host = new ImageConsumer();
        host.readerFactory = file -> new RecordingPreviewReader();
        YoloImageDisplayPanel panel = new YoloImageDisplayPanel();
        SwingUtilities.invokeAndWait(() -> {
            panel.setSizeHintsEnabled(true);
            try { panel.setImageFiles(Arrays.asList(first, second), first, host); }
            catch (IOException error) { throw new AssertionError(error); }
        });
        try {
            awaitPreview(panel, 400);
            javax.swing.JCheckBox shared = (javax.swing.JCheckBox) previewField(panel, "sharedSizeCheckBox");
            JScrollBar z = (JScrollBar) previewField(panel, "zScrollBar");
            JScrollBar time = (JScrollBar) previewField(panel, "timeScrollBar");
            drawReference(panel, 100, 100, 100, 80);
            java.awt.geom.Rectangle2D.Double original = panel.getBoxes().get(0);
            assertTrue(shared.isVisible());
            assertTrue(shared.isSelected());
            assertEquals(panel.getFont(), shared.getFont());
            JButton fileArrow = (JButton) previewField(panel, "previousFileButton");
            assertEquals(fileArrow.getX(), shared.getX());
            assertTrue(shared.getY() >= fileArrow.getY() + fileArrow.getHeight());
            SwingUtilities.invokeAndWait(() -> { z.setValue(3); time.setValue(1); });
            awaitPreview(panel, 400);
            assertEquals(original, panel.getBoxes().get(0));
            SwingUtilities.invokeAndWait(shared::doClick);
            drawReference(panel, 150, 120, 200, 160);
            java.awt.geom.Rectangle2D.Double individualTime = panel.getBoxes().get(0);
            SwingUtilities.invokeAndWait(() -> time.setValue(0));
            awaitPreview(panel, 400);
            assertTrue(shared.isSelected());
            assertEquals(original, panel.getBoxes().get(0));
            drawReference(panel, 100, 100, 140, 110);
            java.awt.geom.Rectangle2D.Double updatedShared = panel.getBoxes().get(0);
            SwingUtilities.invokeAndWait(() -> time.setValue(1));
            awaitPreview(panel, 400);
            assertFalse(shared.isSelected());
            assertEquals(individualTime, panel.getBoxes().get(0));
            JButton next = (JButton) previewField(panel, "nextFileButton");
            JButton previous = (JButton) previewField(panel, "previousFileButton");
            SwingUtilities.invokeAndWait(next::doClick);
            awaitPreview(panel, 400);
            assertTrue(shared.isSelected());
            assertEquals(updatedShared, panel.getBoxes().get(0));
            SwingUtilities.invokeAndWait(shared::doClick);
            drawReference(panel, 100, 100, 60, 50);
            java.awt.geom.Rectangle2D.Double individualFile = panel.getBoxes().get(0);
            io.bioimage.modelrunner.gui.custom.SizeHints.Snapshot snapshot = panel.getSizeHints();
            assertEquals(individualTime, snapshot.forImage(first).apply(1));
            assertEquals(individualFile, snapshot.forImage(second).apply(0));
            assertEquals(updatedShared, snapshot.forImage(second).apply(2));
            SwingUtilities.invokeAndWait(previous::doClick);
            awaitPreview(panel, 400);
            assertEquals(updatedShared, panel.getBoxes().get(0));
            SwingUtilities.invokeAndWait(() -> time.setValue(1));
            awaitPreview(panel, 400);
            assertEquals(individualTime, panel.getBoxes().get(0));
            SwingUtilities.invokeAndWait(shared::doClick);
            assertEquals(updatedShared, panel.getBoxes().get(0));
            assertEquals(individualTime, snapshot.forImage(first).apply(1));
            assertEquals(1, panel.getBoxes().size());
        } finally { SwingUtilities.invokeAndWait(panel::clearImage); }
    }

    @Test
    public void aSingleVolumeKeepsItsLatestRectangleOnEverySliceWithoutASharingCheckbox() throws Exception {
        YoloImageDisplayPanel panel = new YoloImageDisplayPanel();
        SwingUtilities.invokeAndWait(() -> {
            panel.setSizeHintsEnabled(true);
            panel.setImage(ArrayImgs.unsignedBytes(1024, 512, 1, 4, 1), "volume");
        });
        try {
            awaitPreview(panel, 400);
            drawReference(panel, 100, 100, 100, 80);
            java.awt.geom.Rectangle2D.Double first = panel.getBoxes().get(0);
            javax.swing.JCheckBox shared = (javax.swing.JCheckBox) previewField(panel, "sharedSizeCheckBox");
            assertFalse(shared.isVisible());
            JScrollBar z = (JScrollBar) previewField(panel, "zScrollBar");
            SwingUtilities.invokeAndWait(() -> z.setValue(3));
            awaitPreview(panel, 400);
            assertEquals(first, panel.getBoxes().get(0));
            drawReference(panel, 200, 120, 180, 100);
            java.awt.geom.Rectangle2D.Double replacement = panel.getBoxes().get(0);
            assertNotNull(replacement);
            assertFalse(first.equals(replacement));
            SwingUtilities.invokeAndWait(() -> z.setValue(0));
            awaitPreview(panel, 400);
            assertEquals(replacement, panel.getBoxes().get(0));
            assertFalse(shared.isVisible());
            assertEquals(1, panel.getBoxes().size());
        } finally { SwingUtilities.invokeAndWait(panel::clearImage); }
    }

    private static void drawReference(YoloImageDisplayPanel panel, int x, int y, int width, int height) throws Exception {
        java.awt.Rectangle image = (java.awt.Rectangle) previewField(panel, "currentImageRect");
        int x1 = image.x + (int) Math.round(x * image.width / 1024.0);
        int y1 = image.y + (int) Math.round(y * image.height / 512.0);
        int x2 = image.x + (int) Math.round((x + width) * image.width / 1024.0);
        int y2 = image.y + (int) Math.round((y + height) * image.height / 512.0);
        SwingUtilities.invokeAndWait(() -> {
            panel.setDrawEnabled(true);
            for (int kind : new int[] {java.awt.event.MouseEvent.MOUSE_PRESSED,
                    java.awt.event.MouseEvent.MOUSE_DRAGGED, java.awt.event.MouseEvent.MOUSE_RELEASED}) {
                boolean start = kind == java.awt.event.MouseEvent.MOUSE_PRESSED;
                panel.dispatchEvent(new java.awt.event.MouseEvent(panel, kind, System.currentTimeMillis(),
                        0, start ? x1 : x2, start ? y1 : y2, 1, false, java.awt.event.MouseEvent.BUTTON1));
            }
        });
    }

    @Test
    public void outsideClicksCommitThroughTheInstalledListenerAndDetachWithThePanel() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            YoloImageSourcePanel source = new YoloImageSourcePanel();
            java.util.concurrent.atomic.AtomicInteger commits = new java.util.concurrent.atomic.AtomicInteger();
            source.getSystemImagesRadio().setSelected(true);
            source.getSystemPathField().addActionListener(event -> commits.incrementAndGet());
            javax.swing.JPanel background = new javax.swing.JPanel();
            background.setFocusable(false);
            source.addNotify();
            try {
                java.awt.event.FocusEvent gained = new java.awt.event.FocusEvent(
                        source.getSystemPathField(), java.awt.event.FocusEvent.FOCUS_GAINED);
                for (java.awt.event.FocusListener listener : source.getSystemPathField().getFocusListeners())
                    listener.focusGained(gained);
                background.dispatchEvent(new java.awt.event.MouseEvent(background, java.awt.event.MouseEvent.MOUSE_PRESSED,
                        System.currentTimeMillis(), 0, 2, 2, 1, false, java.awt.event.MouseEvent.BUTTON1));
                assertEquals(1, commits.get());
            } finally { source.removeNotify(); }
            // A removed window must no longer observe clicks elsewhere in Icy.
            java.awt.event.FocusEvent gained = new java.awt.event.FocusEvent(
                    source.getSystemPathField(), java.awt.event.FocusEvent.FOCUS_GAINED);
            for (java.awt.event.FocusListener listener : source.getSystemPathField().getFocusListeners())
                listener.focusGained(gained);
            background.dispatchEvent(new java.awt.event.MouseEvent(background, java.awt.event.MouseEvent.MOUSE_PRESSED,
                    System.currentTimeMillis(), 0, 2, 2, 1, false, java.awt.event.MouseEvent.BUTTON1));
            assertEquals(1, commits.get());
        });
    }

    @Test
    public void rapidSliceChangesDoNotInterruptOrPoisonTheSharedDiskReader() throws Exception {
        File file = folder.newFile("many-slices.tif");
        java.util.concurrent.atomic.AtomicBoolean poisoned = new java.util.concurrent.atomic.AtomicBoolean();
        java.util.concurrent.atomic.AtomicReference<CountDownLatch[]> gate = new java.util.concurrent.atomic.AtomicReference<>();
        ImageConsumer host = new ImageConsumer();
        host.readerFactory = ignored -> new PreviewReader() {
            public Info info() { return new Info(32, 32, 1, 80, 1); }
            public BufferedImage read(int z, int t, int level, java.awt.Rectangle region) throws IOException {
                if (poisoned.get()) throw new IOException("Shared decoder was invalidated by interruption");
                CountDownLatch[] blocked = gate.getAndSet(null);
                if (blocked != null) {
                    blocked[0].countDown();
                    try { assertTrue(blocked[1].await(5, TimeUnit.SECONDS)); }
                    catch (InterruptedException error) {
                        poisoned.set(true);
                        Thread.currentThread().interrupt();
                        throw new IOException("Decoder channel closed by interruption", error);
                    }
                }
                BufferedImage image = new BufferedImage(region.width, region.height, BufferedImage.TYPE_INT_RGB);
                java.awt.Graphics2D graphics = image.createGraphics();
                graphics.setColor(new java.awt.Color(z + 1));
                graphics.fillRect(0, 0, image.getWidth(), image.getHeight());
                graphics.dispose();
                return image;
            }
            public void close() {}
        };
        YoloImageDisplayPanel panel = new YoloImageDisplayPanel();
        SwingUtilities.invokeAndWait(() -> {
            try { panel.setImageFile(file, "stack", false, host); }
            catch (IOException error) { throw new AssertionError(error); }
        });
        CountDownLatch[] blocked = null;
        try {
            awaitPreview(panel, 200);
            JScrollBar z = (JScrollBar) previewField(panel, "zScrollBar");
            for (int pass = 0; pass < 2; pass++) {
                blocked = new CountDownLatch[] {new CountDownLatch(1), new CountDownLatch(1)};
                gate.set(blocked);
                int start = 20 + pass * 20, last = 70 + pass;
                SwingUtilities.invokeAndWait(() -> z.setValue(start));
                assertTrue("A disk read must be active before scrolling", blocked[0].await(3, TimeUnit.SECONDS));
                SwingUtilities.invokeAndWait(() -> {
                    z.setValueIsAdjusting(true);
                    for (int i = start; i < last; i++) z.setValue(i);
                    z.setValueIsAdjusting(false);
                    // Arrow auto-repeat does not set valueIsAdjusting.
                    for (int i = 0; i < 20; i++) z.setValue(i + start);
                    z.setValue(last);
                });
                blocked[1].countDown();
                awaitPreview(panel, 200);
                assertFalse("Cancelling stale previews must not interrupt shared decoding", poisoned.get());
                assertNull(previewField(panel, "renderError"));
                assertEquals(last, panel.getPreviewZ());
                BufferedImage image = (BufferedImage) previewField(panel, "renderedViewport");
                assertEquals(last + 1, image.getRGB(0, 0) & 0xffffff);
            }
        } finally {
            if (blocked != null) blocked[1].countDown();
            SwingUtilities.invokeAndWait(panel::clearImage);
        }
    }

    @Test
    public void sourceSelectionAndFocusLossPreserveNavigatedSystemPreviewAcrossAllWorkflows() throws Exception {
        File directory = folder.newFolder("source");
        File first = new File(directory, "a.png"), second = new File(directory, "b.png");
        ImageIO.write(new BufferedImage(2, 2, BufferedImage.TYPE_BYTE_GRAY), "png", first);
        ImageIO.write(new BufferedImage(2, 2, BufferedImage.TYPE_BYTE_GRAY), "png", second);
        List<Function<ImageConsumer, Object>> factories = Arrays.asList(
                c -> new YOLOPluginUI(c, null), c -> new StarDistPluginUI(c, null),
                c -> new UNetPluginUI(c, null), c -> new CrossGoosePluginUI(c, null),
                c -> new CellposePluginUI(c, null));
        for (Function<ImageConsumer, Object> factory : factories) {
            ImageConsumer host = new ImageConsumer() {
                @Override public void setComponents(List<JComponent> components) { componentsGui = components; }
                @Override public void updateGUI() {
                    if (componentsGui.get(1).isEnabled())
                        ((YoloImageDisplayPanel) componentsGui.get(3)).setBufferedImage(
                                new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB), "Host image", false);
                }
            };
            host.image = ArrayImgs.unsignedBytes(2, 2, 3, 1, 1);
            host.readerFactory = file -> new RecordingPreviewReader();
            Object[] ui = new Object[1];
            SwingUtilities.invokeAndWait(() -> ui[0] = factory.apply(host));
            YoloInferencePanel inference = (YoloInferencePanel) ui[0].getClass().getMethod("getInferencePanel").invoke(ui[0]);
            YoloImageSourcePanel source = inference.getImageSourcePanel();
            YoloImageDisplayPanel preview = inference.getImageDisplayPanel();
            try {
                long hostVersion = (Long) previewField(preview, "sourceVersion");
                SwingUtilities.invokeAndWait(() -> {
                    source.getSystemPathField().setText(directory.getAbsolutePath());
                    losePathFocus(source);
                });
                assertEquals("Typing an inactive path must not replace the host preview", hostVersion,
                        previewField(preview, "sourceVersion"));
                SwingUtilities.invokeAndWait(() -> {
                    java.awt.datatransfer.Transferable transfer = new java.awt.datatransfer.Transferable() {
                        public java.awt.datatransfer.DataFlavor[] getTransferDataFlavors() {
                            return new java.awt.datatransfer.DataFlavor[] {java.awt.datatransfer.DataFlavor.javaFileListFlavor};
                        }
                        public boolean isDataFlavorSupported(java.awt.datatransfer.DataFlavor flavor) {
                            return java.awt.datatransfer.DataFlavor.javaFileListFlavor.equals(flavor);
                        }
                        public Object getTransferData(java.awt.datatransfer.DataFlavor flavor) {
                            return java.util.Collections.singletonList(directory);
                        }
                    };
                    assertTrue(source.getSystemPathField().getTransferHandler().importData(
                            new javax.swing.TransferHandler.TransferSupport(source.getSystemPathField(), transfer)));
                    assertTrue(source.getSystemImagesRadio().isSelected());
                });
                awaitPreview(preview, 200);
                JButton previous = (JButton) previewField(preview, "previousFileButton");
                JButton next = (JButton) previewField(preview, "nextFileButton");
                SwingUtilities.invokeAndWait(() -> { if (next.isEnabled()) next.doClick(); else previous.doClick(); });
                awaitPreview(preview, 200);
                JScrollBar z = (JScrollBar) previewField(preview, "zScrollBar");
                JScrollBar time = (JScrollBar) previewField(preview, "timeScrollBar");
                SwingUtilities.invokeAndWait(() -> { z.setValue(3); time.setValue(2); });
                awaitPreview(preview, 200);
                File navigated = preview.getPreviewFile();
                long version = (Long) previewField(preview, "sourceVersion");
                SwingUtilities.invokeAndWait(() -> {
                    losePathFocus(source);
                    source.getSystemPathField().postActionEvent();
                    source.getSystemImagesRadio().doClick();
                });
                assertEquals(version, previewField(preview, "sourceVersion"));
                assertEquals(navigated, preview.getPreviewFile());
                assertEquals(3, preview.getPreviewZ());
                assertEquals(2, preview.getPreviewTime());
                SwingUtilities.invokeAndWait(source.getOpenImagesRadio()::doClick);
                assertNull(preview.getPreviewFile());
                SwingUtilities.invokeAndWait(() -> {
                    source.getSystemImagesRadio().doClick();
                    source.getSystemPathField().setText(first.getAbsolutePath());
                    losePathFocus(source);
                });
                awaitPreview(preview, 200);
                assertEquals("A changed path is accepted without Enter", first, preview.getPreviewFile());
                SwingUtilities.invokeAndWait(() -> {
                    source.getSystemPathField().setText(second.getAbsolutePath());
                    java.awt.event.FocusEvent gained = new java.awt.event.FocusEvent(
                            source.getSystemPathField(), java.awt.event.FocusEvent.FOCUS_GAINED);
                    for (java.awt.event.FocusListener listener : source.getSystemPathField().getFocusListeners())
                        listener.focusGained(gained);
                    clickOutsidePath(source, source.getSystemPathField());
                });
                assertEquals("Clicking inside the editor must not commit it", first, preview.getPreviewFile());
                SwingUtilities.invokeAndWait(() -> {
                    javax.swing.JPanel background = new javax.swing.JPanel();
                    background.setFocusable(false);
                    clickOutsidePath(source, background);
                });
                awaitPreview(preview, 200);
                assertEquals("Clicking a non-focusable background must commit the path", second, preview.getPreviewFile());
            } finally {
                SwingUtilities.invokeAndWait(() -> {
                    try { ui[0].getClass().getMethod("close").invoke(ui[0]); }
                    catch (Exception error) { throw new AssertionError(error); }
                });
            }
        }
    }

    private static void losePathFocus(YoloImageSourcePanel source) {
        java.awt.event.FocusEvent gained = new java.awt.event.FocusEvent(
                source.getSystemPathField(), java.awt.event.FocusEvent.FOCUS_GAINED);
        for (java.awt.event.FocusListener listener : source.getSystemPathField().getFocusListeners()) listener.focusGained(gained);
        java.awt.event.FocusEvent event = new java.awt.event.FocusEvent(
                source.getSystemPathField(), java.awt.event.FocusEvent.FOCUS_LOST);
        for (java.awt.event.FocusListener listener : source.getSystemPathField().getFocusListeners()) listener.focusLost(event);
    }

    private static void clickOutsidePath(YoloImageSourcePanel source, java.awt.Component target) {
        try {
            Method commit = YoloImageSourcePanel.class.getDeclaredMethod("commitPathOnOutsideClick", java.awt.AWTEvent.class);
            commit.setAccessible(true);
            commit.invoke(source, new java.awt.event.MouseEvent(target, java.awt.event.MouseEvent.MOUSE_PRESSED,
                    System.currentTimeMillis(), 0, 2, 2, 1, false, java.awt.event.MouseEvent.BUTTON1));
        } catch (Exception error) { throw new AssertionError(error); }
    }

    @Test
    public void diskPreviewNavigatesZTimeAndExactInferenceFileOrderOffTheEdt() throws Exception {
        File first = folder.newFile("second.tif"), second = folder.newFile("first.tif");
        ImageConsumer host = new ImageConsumer();
        RecordingPreviewReader reader = new RecordingPreviewReader();
        RecordingPreviewReader next = new RecordingPreviewReader();
        host.readerFactory = file -> file.equals(first) ? reader : next;
        YoloImageDisplayPanel panel = new YoloImageDisplayPanel();
        SwingUtilities.invokeAndWait(() -> {
            try { panel.setImageFiles(Arrays.asList(first, second), first, host); }
            catch (IOException error) { throw new AssertionError(error); }
        });
        try {
            awaitPreview(panel, 200);
            JScrollBar z = (JScrollBar) previewField(panel, "zScrollBar");
            JScrollBar time = (JScrollBar) previewField(panel, "timeScrollBar");
            assertTrue(z.isVisible());
            assertTrue(time.isVisible());
            assertEquals(4, z.getMaximum());
            assertEquals(3, time.getMaximum());
            assertTrue(z.getY() < time.getY());
            SwingUtilities.invokeAndWait(() -> { z.setValue(3); time.setValue(2); });
            awaitPreview(panel, 200);
            assertEquals(3, panel.getPreviewZ());
            assertEquals(2, panel.getPreviewTime());
            BufferedImage pixels = (BufferedImage) previewField(panel, "renderedViewport");
            assertEquals(0x302033, pixels.getRGB(20, 20) & 0xffffff);
            JButton previous = (JButton) previewField(panel, "previousFileButton");
            JButton nextButton = (JButton) previewField(panel, "nextFileButton");
            assertFalse(previous.isEnabled());
            SwingUtilities.invokeAndWait(nextButton::doClick);
            awaitPreview(panel, 200);
            assertEquals(second, panel.getPreviewFile());
            assertEquals(0, panel.getPreviewZ());
            assertEquals(0, panel.getPreviewTime());
            assertFalse(nextButton.isEnabled());
            assertTrue(reader.closed.await(3, TimeUnit.SECONDS));
            assertEquals(0, host.calls);
            assertEquals(0, host.previewCalls);
        } finally { SwingUtilities.invokeAndWait(panel::clearImage); }
        assertTrue(next.closed.await(3, TimeUnit.SECONDS));
    }

    @Test
    public void staleMetadataCannotReplaceNewerFileAndItsReaderIsClosed() throws Exception {
        File first = folder.newFile("slow.tif"), second = folder.newFile("new.tif");
        CountDownLatch started = new CountDownLatch(1), release = new CountDownLatch(1);
        RecordingPreviewReader stale = new RecordingPreviewReader(), current = new RecordingPreviewReader();
        ImageConsumer host = new ImageConsumer();
        host.readerFactory = file -> {
            if (file.equals(second)) return current;
            started.countDown();
            boolean waiting = true;
            while (waiting) {
                try { release.await(); waiting = false; }
                catch (InterruptedException ignored) { /* Simulate a decoder that cannot be interrupted. */ }
            }
            return stale;
        };
        YoloImageDisplayPanel panel = new YoloImageDisplayPanel();
        try {
            SwingUtilities.invokeAndWait(() -> {
                try { panel.setImageFile(first, "slow", false, host); }
                catch (IOException error) { throw new AssertionError(error); }
            });
            assertTrue(started.await(3, TimeUnit.SECONDS));
            SwingUtilities.invokeAndWait(() -> {
                try { panel.setImageFile(second, "new", false, host); }
                catch (IOException error) { throw new AssertionError(error); }
            });
            awaitPreview(panel, 200);
            release.countDown();
            assertTrue(stale.closed.await(3, TimeUnit.SECONDS));
            assertEquals(second, panel.getPreviewFile());
            assertNotNull(previewField(panel, "renderedViewport"));
            assertEquals(1, current.closed.getCount());
        } finally {
            release.countDown();
            SwingUtilities.invokeAndWait(panel::clearImage);
        }
        assertTrue(current.closed.await(3, TimeUnit.SECONDS));
    }

    @Test
    public void openImageNavigationPreservesXyCzTAxes() throws Exception {
        RandomAccessibleInterval<net.imglib2.type.numeric.integer.UnsignedByteType> image =
                ArrayImgs.unsignedBytes(3, 3, 3, 4, 3);
        net.imglib2.RandomAccess<net.imglib2.type.numeric.integer.UnsignedByteType> access = image.randomAccess();
        for (int t = 0; t < 3; t++) for (int z = 0; z < 4; z++)
            for (int c = 0; c < 3; c++) for (int y = 0; y < 3; y++) for (int x = 0; x < 3; x++) {
                access.setPosition(new long[] {x, y, c, z, t});
                access.get().set(x == 0 && y == 0 ? 0 : c == 0 ? z * 50 : c == 1 ? t * 50 : 255);
            }
        YoloImageDisplayPanel panel = new YoloImageDisplayPanel();
        try {
            SwingUtilities.invokeAndWait(() -> panel.setImage(image, "open stack"));
            awaitPreview(panel, 200);
            JScrollBar z = (JScrollBar) previewField(panel, "zScrollBar");
            JScrollBar time = (JScrollBar) previewField(panel, "timeScrollBar");
            SwingUtilities.invokeAndWait(() -> { z.setValue(2); time.setValue(1); });
            awaitPreview(panel, 200);
            BufferedImage pixels = (BufferedImage) previewField(panel, "renderedViewport");
            assertEquals(0x6432ff, pixels.getRGB(pixels.getWidth() - 2, pixels.getHeight() - 2) & 0xffffff);
            assertEquals(2, panel.getPreviewZ());
            assertEquals(1, panel.getPreviewTime());
        } finally { SwingUtilities.invokeAndWait(panel::clearImage); }
    }

    @Test
    public void hostImageReaderExposesOneStepArrowsAndDoesNotLoadAWholeStack() throws Exception {
        RecordingPreviewReader reader = new RecordingPreviewReader(new PreviewReader.Info(1024, 512, 1, 4, 3, 0));
        YoloImageDisplayPanel panel = new YoloImageDisplayPanel();
        try {
            SwingUtilities.invokeAndWait(() -> panel.setImageReader(reader, "Host grayscale Z/T stack"));
            awaitPreview(panel, 200);
            JScrollBar z = (JScrollBar) previewField(panel, "zScrollBar");
            JScrollBar time = (JScrollBar) previewField(panel, "timeScrollBar");
            assertTrue(z.isVisible());
            assertTrue(time.isVisible());
            assertEquals(4, z.getMaximum());
            assertEquals(3, time.getMaximum());
            // Reinstalling a host theme must not remove the arrows.
            SwingUtilities.invokeAndWait(() -> { z.updateUI(); time.updateUI(); panel.doLayout(); });
            clickArrow(z, javax.swing.SwingConstants.EAST);
            clickArrow(time, javax.swing.SwingConstants.EAST);
            awaitPreview(panel, 200);
            assertEquals(1, panel.getPreviewZ());
            assertEquals(1, panel.getPreviewTime());
            BufferedImage pixels = (BufferedImage) previewField(panel, "renderedViewport");
            assertEquals(0x101033, pixels.getRGB(20, 20) & 0xffffff);
            clickArrow(z, javax.swing.SwingConstants.WEST);
            clickArrow(time, javax.swing.SwingConstants.WEST);
            awaitPreview(panel, 200);
            assertEquals(0, panel.getPreviewZ());
            assertEquals(0, panel.getPreviewTime());
            clickArrow(z, javax.swing.SwingConstants.WEST);
            assertEquals(0, panel.getPreviewZ());
        } finally { SwingUtilities.invokeAndWait(panel::clearImage); }
        assertTrue(reader.closed.await(3, TimeUnit.SECONDS));
    }

    private static void clickArrow(JScrollBar bar, int direction) throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            bar.doLayout();
            for (java.awt.Component component : bar.getComponents()) {
                if (!(component instanceof javax.swing.plaf.basic.BasicArrowButton)) continue;
                javax.swing.plaf.basic.BasicArrowButton button = (javax.swing.plaf.basic.BasicArrowButton) component;
                if (button.getDirection() != direction) continue;
                assertTrue(button.isVisible());
                assertTrue(button.getWidth() > 0 && button.getHeight() > 0);
                long when = System.currentTimeMillis();
                button.dispatchEvent(new java.awt.event.MouseEvent(button, java.awt.event.MouseEvent.MOUSE_PRESSED,
                        when, 0, 2, 2, 1, false, java.awt.event.MouseEvent.BUTTON1));
                button.dispatchEvent(new java.awt.event.MouseEvent(button, java.awt.event.MouseEvent.MOUSE_RELEASED,
                        when, 0, 2, 2, 1, false, java.awt.event.MouseEvent.BUTTON1));
                return;
            }
            fail("Missing one-step navigation arrow");
        });
    }

    private static final class RecordingPreviewReader implements PreviewReader {
        final CountDownLatch closed = new CountDownLatch(1);
        private final Info geometry;
        RecordingPreviewReader() { this(new Info(1024, 512, 3, 4, 3, 0, 1, 2)); }
        RecordingPreviewReader(Info geometry) { this.geometry = geometry; }
        @Override public Info info() { return geometry; }
        @Override public BufferedImage read(int z, int t, int level, java.awt.Rectangle region) {
            assertFalse("Decode regions off the EDT", SwingUtilities.isEventDispatchThread());
            assertTrue(region.x >= 0 && region.x + region.width <= info().width);
            assertTrue(region.y >= 0 && region.y + region.height <= info().height);
            assertTrue(region.width <= 256 * (1L << level));
            BufferedImage image = new BufferedImage((region.width + (1 << level) - 1) >> level,
                    (region.height + (1 << level) - 1) >> level, BufferedImage.TYPE_INT_RGB);
            java.awt.Graphics2D graphics = image.createGraphics();
            graphics.setColor(new java.awt.Color((z * 16 << 16) | (t * 16 << 8) | 0x33));
            graphics.fillRect(0, 0, image.getWidth(), image.getHeight());
            graphics.dispose();
            return image;
        }
        @Override public void close() { closed.countDown(); }
    }

    @Test
    public void imageJLookupTableFailureUsesOneCachedHostPreviewOffTheEdt() throws Exception {
        File file = imageJLookupTableTiff();
        ImageConsumer consumer = new ImageConsumer();
        consumer.image = ArrayImgs.unsignedShorts(new short[] {68, 886, 300, 500}, 2, 2);
        YoloImageDisplayPanel panel = new YoloImageDisplayPanel();
        SwingUtilities.invokeAndWait(() -> {
            try { panel.setImageFile(file, "", false, consumer); }
            catch (IOException e) { throw new AssertionError(e); }
        });
        awaitPreview(panel, 200);
        assertNotNull(previewField(panel, "renderedViewport"));
        assertNull(previewField(panel, "renderError"));
        assertEquals(1, consumer.previewCalls);
        assertEquals(0, consumer.calls);
        BufferedImage image = (BufferedImage) previewField(panel, "renderedViewport");
        assertEquals(0, image.getRGB(0, 0) & 255);
        assertEquals(255, image.getRGB(image.getWidth() - 1, 0) & 255);
        awaitPreview(panel, 300);
        assertEquals("Zoom/resize must reuse decoded host pixels", 1, consumer.previewCalls);
        SwingUtilities.invokeAndWait(panel::clearImage);
    }

    @Test
    public void failedPreviewShowsAnErrorInsteadOfRetryingForever() throws Exception {
        File file = imageJLookupTableTiff();
        YoloImageDisplayPanel panel = new YoloImageDisplayPanel();
        SwingUtilities.invokeAndWait(() -> {
            try { panel.setImageFile(file, "", false); }
            catch (IOException e) { throw new AssertionError(e); }
        });
        awaitPreview(panel, 200);
        assertNotNull(previewField(panel, "renderError"));
        assertNull(previewField(panel, "pendingRequest"));
        long version = (Long) previewField(panel, "renderVersion");
        awaitPreview(panel, 200);
        assertEquals(version, previewField(panel, "renderVersion"));
        SwingUtilities.invokeAndWait(panel::clearImage);
    }

    private File imageJLookupTableTiff() throws IOException {
        File file = folder.newFile("imagej-lut.tif");
        // ImageJ's 256-entry display LUT on a 16-bit grayscale TIFF. ImageIO's
        // writer removes this combination, so construct the small fixture directly.
        int pixels = 134, lut = pixels + 8;
        ByteBuffer data = ByteBuffer.allocate(lut + 768 * 2).order(ByteOrder.LITTLE_ENDIAN);
        data.put((byte) 'I').put((byte) 'I').putShort((short) 42).putInt(8).putShort((short) 10);
        int[][] fields = {{256, 4, 1, 2}, {257, 4, 1, 2}, {258, 3, 1, 16}, {259, 3, 1, 1},
                {262, 3, 1, 1}, {273, 4, 1, pixels}, {277, 3, 1, 1}, {278, 4, 1, 2},
                {279, 4, 1, 8}, {320, 3, 768, lut}};
        for (int[] field : fields)
            data.putShort((short) field[0]).putShort((short) field[1]).putInt(field[2]).putInt(field[3]);
        data.putInt(0).putShort((short) 68).putShort((short) 886).putShort((short) 300).putShort((short) 500);
        for (int i = 0; i < 768; i++) data.putShort((short) ((i % 256) * 256));
        Files.write(file.toPath(), data.array());
        return file;
    }

    private static void awaitPreview(YoloImageDisplayPanel panel, int size) throws Exception {
        long deadline = System.nanoTime() + 5_000_000_000L;
        do {
            SwingUtilities.invokeAndWait(() -> {
                panel.setSize(size, size);
                panel.doLayout();
                BufferedImage canvas = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
                java.awt.Graphics2D graphics = canvas.createGraphics();
                panel.paint(graphics);
                graphics.dispose();
            });
            if (previewField(panel, "pendingRequest") == null
                    && (previewField(panel, "renderedViewport") != null || previewField(panel, "renderError") != null)) return;
            Thread.sleep(20);
        } while (System.nanoTime() < deadline);
        fail("Preview worker never completed");
    }

    private static Object previewField(YoloImageDisplayPanel panel, String name) throws Exception {
        java.lang.reflect.Field field = YoloImageDisplayPanel.class.getDeclaredField(name);
        field.setAccessible(true);
        Object[] value = new Object[1];
        SwingUtilities.invokeAndWait(() -> {
            try { value[0] = field.get(panel); }
            catch (IllegalAccessException e) { throw new AssertionError(e); }
        });
        return value[0];
    }

    @Test
    public void inferenceReadersDelegateWithoutConvertingPixelsOrDroppingCztAxes() throws Exception {
        ImageConsumer consumer = new ImageConsumer();
        Object[] interfaces = new Object[5];
        SwingUtilities.invokeAndWait(() -> {
            interfaces[0] = new CellposePluginUI(consumer, null);
            interfaces[1] = new StarDistPluginUI(consumer, null);
            interfaces[2] = new UNetPluginUI(consumer, null);
            interfaces[3] = new CrossGoosePluginUI(consumer, null);
            interfaces[4] = new YOLOPluginUI(consumer, null);
        });
        try {
            for (Object ui : interfaces) {
                Class<?> owner = ui instanceof CrossGoosePluginUI ? UNetPluginUI.class : ui.getClass();
                Method read = owner.getDeclaredMethod("readImageFileAsRai", File.class);
                read.setAccessible(true);
                File source = new File(folder.getRoot(), "host-decoded.ome.tif");
                for (RandomAccessibleInterval<?> image : new RandomAccessibleInterval<?>[] {
                        ArrayImgs.unsignedShorts(new short[] {1000, 1001, 2000, 2001, 3000, 3001}, 2, 1, 3, 1, 1),
                        ArrayImgs.doubles(3, 2, 4, 5, 6)}) {
                    consumer.image = image;
                    int previous = consumer.calls;
                    assertSame(image, read.invoke(ui, source));
                    assertSame(source, consumer.file);
                    assertEquals(previous + 1, consumer.calls);
                    assertEquals(image.dimension(2), consumer.getImageFileChannels(source));
                }
                consumer.error = new IOException("Host could not decode this image");
                try {
                    read.invoke(ui, source);
                    fail("Host read failure must not trigger a fallback RGB reader");
                } catch (InvocationTargetException error) {
                    assertSame(consumer.error, error.getCause());
                } finally {
                    consumer.error = null;
                }
            }
        } finally {
            for (Object ui : interfaces) ui.getClass().getMethod("close").invoke(ui);
        }
    }

    @Test
    public void trainingMaskReadersPreserveSixteenBitLabelsWithWrongExtensions() throws Exception {
        for (String format : new String[] {"png", "tiff"}) {
            File mask = new File(folder.getRoot(), format + "-mask." + (format.equals("png") ? "tif" : "png"));
            BufferedImage labels = new BufferedImage(12, 11, BufferedImage.TYPE_USHORT_GRAY);
            labels.getRaster().setSample(1, 1, 0, 1);
            labels.getRaster().setSample(3, 3, 0, 257);
            labels.getRaster().setSample(6, 6, 0, 65535);
            assertTrue(ImageIO.write(labels, format, mask));
            Method stats = SegmentationDatasetPreparer.class.getDeclaredMethod("readMaskStats", File.class);
            stats.setAccessible(true);
            Object result = stats.invoke(null, mask);
            java.lang.reflect.Field count = result.getClass().getDeclaredField("objectCount");
            count.setAccessible(true);
            assertEquals(3, count.getInt(result));
            Method boxes = Class.forName("io.bioimage.modelrunner.gui.custom.yolo.YoloDatasetPreparer")
                    .getDeclaredMethod("readMaskSample", File.class, File.class);
            boxes.setAccessible(true);
            Object sample = boxes.invoke(null, mask, mask);
            java.lang.reflect.Field objects = sample.getClass().getDeclaredField("boxes");
            objects.setAccessible(true);
            assertEquals(3, ((java.util.List<?>) objects.get(sample)).size());
        }
    }

    private class ImageConsumer extends ConsumerInterface {
        RandomAccessibleInterval<?> image;
        File file;
        IOException error;
        int calls;
        int previewCalls;
        Function<File, PreviewReader> readerFactory;

        @Override public PreviewReader openPreviewReader(File source) {
            assertFalse("Inspect metadata off the EDT", SwingUtilities.isEventDispatchThread());
            return readerFactory == null ? null : readerFactory.apply(source);
        }

        @Override @SuppressWarnings("unchecked")
        public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> openImagePreviewAsRai(File source)
                throws IOException {
            assertFalse("Decode previews off the EDT", SwingUtilities.isEventDispatchThread());
            previewCalls++;
            if (error != null) throw error;
            return (RandomAccessibleInterval<T>) image;
        }

        @Override @SuppressWarnings("unchecked")
        public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> openImageAsRai(File source)
                throws IOException {
            file = source;
            calls++;
            if (error != null) throw error;
            return (RandomAccessibleInterval<T>) image;
        }

        @Override public String getModelsDir() { return folder.getRoot().getAbsolutePath(); }
        @Override public void updateGUI() {}
        @Override public void setComponents(List<JComponent> components) {}
        @Override public void setVarNames(List<String> names) {}
        @Override public void notifyParams(LinkedHashMap<String, String> args) {}
        @Override public Object getFocusedImage() { return null; }
        @Override public String getFocusedImageName() { return null; }
        @Override public Integer getFocusedImageChannels() { return 1; }
        @Override public Integer getFocusedImageSlices() { return 1; }
        @Override public Integer getFocusedImageFrames() { return 1; }
        @Override public Integer getFocusedImageWidth() { return 1; }
        @Override public Integer getFocusedImageHeight() { return 1; }
        @Override public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> getFocusedImageAsRai() { return null; }
        @Override public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> convertIntoRai(Object image) { return null; }
        @Override public <T extends RealType<T> & NativeType<T>> void displayImage(RandomAccessibleInterval<T> image, String axes, String name) {}
        @Override public void displayDetections(List<Detection> detections) {}
    }
}
