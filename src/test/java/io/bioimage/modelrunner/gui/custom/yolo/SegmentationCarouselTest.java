package io.bioimage.modelrunner.gui.custom.yolo;

import static org.junit.Assert.*;
import java.awt.Component;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.prefs.Preferences;
import javax.swing.JButton;
import javax.swing.SwingUtilities;
import org.junit.Test;
import io.bioimage.modelrunner.gui.custom.segmentation.SegmentationOutput;

public class SegmentationCarouselTest {
    @Test public void browsingIsSeparateFromRunningAndChoicesFollowTheirSource() throws Exception {
        Preferences prefs = Preferences.userRoot().node("jdll-carousel-test/" + UUID.randomUUID());
        try {
            SwingUtilities.invokeAndWait(() -> {
                YoloActionPanel actions = new YoloActionPanel();
                YoloImageSourcePanel source = new YoloImageSourcePanel();
                actions.setSize(400,32);
                actions.enableSegmentationOutputs(source,prefs);
                AtomicInteger runs = new AtomicInteger();
                actions.getRunButton().addActionListener(e -> runs.incrementAndGet());
                JButton next = arrow(actions,"Next output"), prev = arrow(actions,"Previous output");
                assertEquals(SegmentationOutput.ROIS, actions.output());
                actions.setRunAvailable(false);
                assertTrue(next.isVisible() && next.isEnabled());
                assertFalse(actions.getRunButton().isEnabled());
                prev.doClick();
                assertEquals(SegmentationOutput.BOTH, actions.output());
                actions.doLayout();
                assertEquals("Run \u2192 ROIs + labels", actions.getRunButton().getText());
                source.getSystemImagesRadio().setSelected(true);
                assertEquals(SegmentationOutput.LABELS, actions.output());
                next.doClick();
                next.doClick();
                assertEquals(SegmentationOutput.ROIS, actions.output());
                source.getOpenImagesRadio().setSelected(true);
                assertEquals(SegmentationOutput.BOTH, actions.output());
                assertEquals(0, runs.get());
                actions.setRunAvailable(true);
                actions.getRunButton().doClick();
                assertEquals(1, runs.get());
                actions.setOutputLocked(true);
                actions.setRunAvailable(true);
                assertFalse(next.isEnabled());
                assertFalse(actions.getRunButton().isEnabled());

                YoloActionPanel reopened = new YoloActionPanel();
                reopened.enableSegmentationOutputs(source,prefs);
                assertEquals(SegmentationOutput.BOTH, reopened.output());
                reopened.setOpenImageRoisSupported(false);
                reopened.setRunAvailable(true);
                assertEquals(SegmentationOutput.LABELS, reopened.output());
                arrow(reopened,"Next output").doClick();
                assertFalse(reopened.getRunButton().isEnabled());
                assertTrue(arrow(reopened,"Previous output").isEnabled());
                source.getSystemImagesRadio().setSelected(true);
                assertEquals(SegmentationOutput.ROIS, reopened.output());
                assertTrue(reopened.getRunButton().isEnabled());
            });
        } finally { prefs.removeNode(); }
    }

    private static JButton arrow(YoloActionPanel p, String name) {
        for (Component c : p.getComponents()) if (c instanceof JButton &&
                name.equals(c.getAccessibleContext().getAccessibleName())) return (JButton)c;
        throw new AssertionError(name);
    }
}
