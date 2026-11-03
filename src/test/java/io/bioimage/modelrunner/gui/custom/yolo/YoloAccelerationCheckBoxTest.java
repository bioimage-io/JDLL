package io.bioimage.modelrunner.gui.custom.yolo;

import static org.junit.Assert.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;
import javax.swing.SwingUtilities;
import org.junit.Test;

public class YoloAccelerationCheckBoxTest {
    private YoloAccelerationCheckBox checkbox(Supplier<Boolean> probe, CountDownLatch finished) throws Exception {
        AtomicReference<YoloAccelerationCheckBox> result = new AtomicReference<>();
        SwingUtilities.invokeAndWait(() -> {
            YoloAccelerationCheckBox box = new YoloAccelerationCheckBox(probe, "CUDA");
            box.addPropertyChangeListener("ToolTipText", event -> finished.countDown());
            result.set(box);
        });
        return result.get();
    }

    private void await(CountDownLatch finished) throws Exception {
        assertTrue("Availability check did not finish", finished.await(3, TimeUnit.SECONDS));
        SwingUtilities.invokeAndWait(() -> {});
    }

    @Test public void unavailableGpuCannotBeReenabledAfterInferenceOrMethodChange() throws Exception {
        CountDownLatch done = new CountDownLatch(1);
        YoloAccelerationCheckBox box = checkbox(() -> false, done);
        await(done);
        SwingUtilities.invokeAndWait(() -> {
            assertFalse(box.isAccelerationAvailable());
            box.setEnabled(false);
            box.setEnabled(true);
            box.setSelected(true);
            assertFalse(box.isEnabled());
            assertFalse(box.isSelected());
            assertTrue(box.getToolTipText().contains("unavailable"));
        });
    }

    @Test public void availableGpuKeepsSelectionWhileTemporarilyLocked() throws Exception {
        CountDownLatch done = new CountDownLatch(1);
        YoloAccelerationCheckBox box = checkbox(() -> true, done);
        await(done);
        SwingUtilities.invokeAndWait(() -> {
            assertTrue(box.isEnabled());
            box.setSelected(true);
            box.setEnabled(false);
            assertFalse(box.isEnabled());
            assertTrue(box.isSelected());
            box.setEnabled(true);
            assertTrue(box.isEnabled());
            assertTrue(box.isSelected());
        });
    }

    @Test public void completingProbeDoesNotOverrideAnActiveRunLock() throws Exception {
        CountDownLatch release = new CountDownLatch(1), done = new CountDownLatch(1);
        YoloAccelerationCheckBox box = checkbox(() -> {
            try { release.await(3, TimeUnit.SECONDS); }
            catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            return true;
        }, done);
        SwingUtilities.invokeAndWait(() -> box.setEnabled(false));
        release.countDown();
        await(done);
        SwingUtilities.invokeAndWait(() -> {
            assertTrue(box.isAccelerationAvailable());
            assertFalse(box.isEnabled());
            box.setEnabled(true);
            assertTrue(box.isEnabled());
        });
    }

    @Test public void failedNativeProbeLeavesAccelerationDisabled() throws Exception {
        CountDownLatch done = new CountDownLatch(1);
        YoloAccelerationCheckBox box = checkbox(() -> { throw new UnsatisfiedLinkError("No driver"); }, done);
        await(done);
        SwingUtilities.invokeAndWait(() -> {
            box.setEnabled(true);
            assertFalse(box.isEnabled());
        });
    }
}
