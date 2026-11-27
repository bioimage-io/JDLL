package io.bioimage.modelrunner.model.special.cellpose;

import java.io.File;
import java.util.Objects;

/** Immutable settings shared by CPU and accelerated training. A null base model requests scratch training. */
public final class CellposeTrainingRequest {
    public final String name;
    public final File dataset;
    public final int epochs;
    public final String device;
    public final String baseModel;
    private final int[] channels;

    /** Base models can be official aliases, native weights or a previously published training directory. */
    public CellposeTrainingRequest(String name, File dataset, int epochs, int[] channels, String device, String baseModel) {
        this.name = Objects.requireNonNull(name, "model name").trim();
        this.dataset = Objects.requireNonNull(dataset, "dataset").getAbsoluteFile();
        if (this.name.isEmpty() || !this.dataset.isDirectory())
            throw new IllegalArgumentException("Provide a model name and a training dataset directory.");
        if (epochs < 1 || epochs == 1 && io.bioimage.modelrunner.gui.custom.training.TrainingContinuation.find(baseModel) == null) throw new IllegalArgumentException("Cellpose requires at least two epochs, including warm-up.");
        if (channels == null || channels.length != 2 || channels[0] < 0 || channels[1] < 0)
            throw new IllegalArgumentException("Provide two nonnegative source channel indices.");
        if (!"cpu".equals(device) && !"cuda".equals(device) && !"mps".equals(device))
            throw new IllegalArgumentException("Choose cpu, cuda or mps.");
        this.epochs = epochs;
        this.channels = channels.clone();
        this.device = device;
        this.baseModel = baseModel;
    }

    public int[] channels() { return channels.clone(); }
}
