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
import io.bioimage.modelrunner.model.detection.Detection;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/** Test the actual disk-entry readers without constructing a GUI or loading a model. */
public class ImageReaderCompatibilityTest {
    @Rule public TemporaryFolder folder = new TemporaryFolder();

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
