package io.bioimage.modelrunner.gui.custom.yolo;

import static org.junit.Assert.*;
import java.awt.Component;
import java.awt.Container;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
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
                assertEquals(SegmentationOutput.BOTH, actions.output());
                assertEquals("Generate labels + ROIs", actions.getRunButton().getText());
                actions.setRunAvailable(false);
                assertTrue(next.isVisible() && next.isEnabled());
                assertFalse(actions.getRunButton().isEnabled());
                prev.doClick();
                assertEquals(SegmentationOutput.LABELS, actions.output());
                assertEquals("Generate labels", actions.getRunButton().getText());
                source.getSystemImagesRadio().setSelected(true);
                assertEquals(SegmentationOutput.BOTH, actions.output());
                next.doClick();
                assertEquals(SegmentationOutput.ROIS, actions.output());
                next.doClick();
                assertEquals(SegmentationOutput.LABELS, actions.output());
                prev.doClick();
                assertEquals("Generate ROIs", actions.getRunButton().getText());
                source.getOpenImagesRadio().setSelected(true);
                assertEquals(SegmentationOutput.LABELS, actions.output());
                assertEquals(0, runs.get());
                actions.setRunAvailable(true);
                actions.getRunButton().doClick();
                assertEquals(1, runs.get());
                actions.setOutputLocked(true);
                actions.setRunAvailable(true);
                assertFalse(next.isEnabled());
                assertFalse(prev.isEnabled());
                assertFalse(actions.getRunButton().isEnabled());
                actions.setOutputLocked(false);
                assertTrue(actions.getRunButton().isEnabled());

                YoloActionPanel reopened = new YoloActionPanel();
                reopened.enableSegmentationOutputs(source,prefs);
                assertEquals(SegmentationOutput.LABELS, reopened.output());
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

    @Test public void positionBarsAndHoverStayInsideOneFixedButton() throws Exception {
        Preferences prefs = Preferences.userRoot().node("jdll-carousel-test/" + UUID.randomUUID());
        try {
            SwingUtilities.invokeAndWait(() -> {
                YoloActionPanel actions = new YoloActionPanel();
                actions.setSize(480, 28);
                actions.enableSegmentationOutputs(new YoloImageSourcePanel(), prefs);
                JButton run = actions.getRunButton();
                Container carousel = run.getParent();
                JButton previous = arrow(actions, "Previous output");
                JButton next = arrow(actions, "Next output");
                Rectangle runArea = run.getBounds();
                Rectangle prevArea = previous.getBounds();
                Rectangle nextArea = next.getBounds();
                assertEquals(actions.getCancelButton().getHeight(), carousel.getHeight());
                assertEquals(actions.getCancelButton().getFont(), run.getFont());
                assertEquals(runArea.x * 2 + runArea.width, carousel.getWidth());

                BufferedImage idle = render(carousel);
                int y = runArea.y + runArea.height / 2;
                int background = idle.getRGB(runArea.x + 1, y);
                int first = -1;
                for (int x = nextArea.x - run.getInsets().right; x < nextArea.x; x++) {
                    if (idle.getRGB(x, y) != background) { first = x; break; }
                }
                assertTrue("Position bars visible before right chevron", first >= 0);
                int active = idle.getRGB(first + 6, y);
                int inactive = idle.getRGB(first, y);
                assertNotEquals(active, inactive);
                assertEquals(inactive, idle.getRGB(first + 12, y));
                assertEquals(background, idle.getRGB(first, y - 5));
                assertEquals(active, idle.getRGB(first + 6, y - 5));
                assertSame(run, SwingUtilities.getDeepestComponentAt(carousel, first, y));

                int sampleY = runArea.y + 2;
                for (int x = prevArea.x; x < nextArea.x + nextArea.width; x++)
                    assertEquals("No internal border", background, idle.getRGB(x, sampleY));
                for (JButton area : new JButton[] {previous, run, next}) {
                    area.getModel().setRollover(true);
                    BufferedImage hovered = render(carousel);
                    for (JButton sample : new JButton[] {previous, run, next}) {
                        int color = hovered.getRGB(sample.getX() + 1, sampleY);
                        if (sample == area) assertNotEquals(background, color);
                        else assertEquals(background, color);
                    }
                    area.getModel().setRollover(false);
                }
                next.doClick(0);
                BufferedImage rois = render(carousel);
                assertEquals("Generate ROIs", run.getText());
                assertEquals(active, rois.getRGB(first + 12, y - 5));
                assertEquals(inactive, rois.getRGB(first + 6, y));
                previous.doClick(0);
                previous.doClick(0);
                assertEquals("Generate labels", run.getText());
                assertEquals(active, render(carousel).getRGB(first, y - 5));
                assertEquals(runArea, run.getBounds());
                assertEquals(prevArea, previous.getBounds());
                assertEquals(nextArea, next.getBounds());
                actions.setSize(320, 28);
                actions.doLayout();
                next.doClick(0);
                assertTrue("Generate both".equals(run.getText()) || "Generate labels + ROIs".equals(run.getText()));
                assertEquals("Generate labels + ROIs", run.getAccessibleContext().getAccessibleName());
            });
        } finally { prefs.removeNode(); }
    }

    private static BufferedImage render(Container carousel) {
        BufferedImage img = new BufferedImage(carousel.getWidth(), carousel.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = img.createGraphics();
        try { carousel.printAll(graphics); } finally { graphics.dispose(); }
        return img;
    }

    private static JButton arrow(YoloActionPanel p, String name) {
        for (Component c : p.getRunButton().getParent().getComponents()) if (c instanceof JButton &&
                name.equals(c.getAccessibleContext().getAccessibleName())) return (JButton)c;
        throw new AssertionError(name);
    }
}
