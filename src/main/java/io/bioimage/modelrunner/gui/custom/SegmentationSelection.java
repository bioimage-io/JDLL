package io.bioimage.modelrunner.gui.custom;

import java.io.File;
import io.bioimage.modelrunner.gui.custom.segmentation.SegmentationOutput;
import io.bioimage.modelrunner.gui.custom.yolo.YoloImageSelectionEntry;
import io.bioimage.modelrunner.gui.custom.yolo.YoloInferencePanel;

/** Captured by the Run action before a worker can observe a change of selection or focus. */
final class SegmentationSelection {
    final boolean files;
    final File source;
    final Object image;
    final String model;
    final SegmentationOutput output;

    SegmentationSelection(YoloInferencePanel panel, File source) {
        files = panel.getImageSourcePanel().getSystemImagesRadio().isSelected();
        this.source = source;
        Object entry = panel.getImageSourcePanel().getOpenImagesComboBox().getSelectedItem();
        image = entry instanceof YoloImageSelectionEntry ? ((YoloImageSelectionEntry)entry).getImage() : null;
        model = panel.getModelSelectionPanel().getSelectedModelValue();
        output = panel.getActionPanel().output();
    }
}
