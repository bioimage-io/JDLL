package io.bioimage.modelrunner.gui.custom.cellpose;

import static org.junit.Assert.*;
import javax.swing.JComboBox;
import javax.swing.SwingUtilities;
import org.junit.Test;
import io.bioimage.modelrunner.gui.custom.ConsumerInterface.ImageChannels;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.view.Views;

public class CellposeChannelsTest {
    @Test public void sourceAndModelDetermineChoicesAndPackedPixels() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            CellposeInferencePanel panel = new CellposeInferencePanel();
            CellposeOptionsPanel options = panel.getOptionsPanel();
            JComboBox<String> primary = options.getCytoplasmComboBox(), secondary = options.getNucleiComboBox();
            assertItems(primary, "gray");
            assertItems(secondary, "none", "gray");
            assertArrayEquals(new int[]{1, 0}, options.selectedChannels());
            secondary.setSelectedItem("gray");
            assertArrayEquals(new int[]{1, 1}, options.selectedChannels());

            CellposeChannels.configure(primary, secondary, new ImageChannels(3, true));
            assertItems(primary, "red", "green", "blue");
            assertItems(secondary, "none", "red", "green", "blue");
            options.selectSavedChannels(new int[]{2, 3});
            assertEquals("green", primary.getSelectedItem());
            assertEquals("blue", secondary.getSelectedItem());
            float[] packed = new float[3];
            int i = 0;
            for (net.imglib2.type.numeric.real.FloatType pixel : Views.flatIterable(CellposeChannels.pack(
                    ArrayImgs.floats(new float[]{11, 22, 33}, 1, 1, 3), options.selectedChannels())))
                packed[i++] = pixel.get();
            assertArrayEquals(new float[]{22, 33, 0}, packed, 0);

            CellposeChannels.configure(primary, secondary, new ImageChannels(3, false));
            assertItems(primary, "channel 1", "channel 2", "channel 3");
            assertItems(secondary, "none", "channel 1", "channel 2", "channel 3");
            assertArrayEquals(new int[]{2, 3}, options.selectedChannels());
            panel.getModelSelectionPanel().addOrSelectModel("nuclei", "nuclei");
            assertFalse(secondary.isVisible());
            assertArrayEquals(new int[]{2, 0}, options.selectedChannels());
            panel.getModelSelectionPanel().addOrSelectModel("cyto3", "cyto3");
            assertTrue(secondary.isVisible());
            assertArrayEquals(new int[]{2, 3}, options.selectedChannels());

            CellposeChannels.configure(primary, secondary, 2);
            assertItems(primary, "channel 1", "channel 2");
            assertItems(secondary, "none", "channel 1", "channel 2");
            assertArrayEquals(new int[]{2, 0}, options.selectedChannels());
            options.selectSavedChannels(new int[]{0, 0});
            assertArrayEquals(new int[]{1, 0}, options.selectedChannels());
        });
    }

    private static void assertItems(JComboBox<String> combo, String... expected) {
        assertEquals(expected.length, combo.getItemCount());
        for (int i = 0; i < expected.length; i++) assertEquals(expected[i], combo.getItemAt(i));
    }
}
