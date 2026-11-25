package io.bioimage.modelrunner.gui.custom.segmentation;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

import io.bioimage.modelrunner.gui.custom.ConsumerInterface;
import io.bioimage.modelrunner.gui.custom.InferenceRunFiles;
import io.bioimage.modelrunner.gui.custom.LabelMaskWriter;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/** Delivery of the final segmentation tensor. Other model outputs have their own delivery path. */
public enum SegmentationOutput {
    ROIS("ROIs", "Generate ROIs", true, false),
    LABELS("Label image", "Generate labels", false, true),
    BOTH("ROIs + labels", "Generate labels + ROIs", true, true);

    public final String title;
    public final String actionLabel;
    private final boolean rois, labels;
    SegmentationOutput(String title, String actionLabel, boolean rois, boolean labels) {
        this.title = title;
        this.actionLabel = actionLabel;
        this.rois = rois;
        this.labels = labels;
    }

    public <T extends RealType<T> & NativeType<T>> void display(ConsumerInterface host, Object source,
            Tensor<T> tensor, String name, SegmentationRegions.Kind kind, BooleanSupplier cancelled,
            Consumer<String> log) throws IOException, InterruptedException {
        SegmentationRegions.check(cancelled);
        SegmentationRegions regions = rois ? SegmentationRegions.fromLabels(tensor, kind, cancelled) : null;
        SegmentationRegions.check(cancelled);
        host.displaySegmentation(source, regions, labels ? tensor : null, name, cancelled);
        if (regions != null && !cancelled.getAsBoolean())
            log.accept("Added " + regions.objects().size() + " ROI(s) to the source image.");
    }

    public <T extends RealType<T> & NativeType<T>> void save(ConsumerInterface host, InferenceRunFiles run,
            File source, Tensor<T> tensor, SegmentationRegions.Kind kind, BooleanSupplier cancelled,
            Consumer<String> log) throws IOException {
        SegmentationRegions.check(cancelled);
        Path directory = run.getDirectory().toPath();
        Path stage = Files.createTempDirectory(directory, ".segmentation-");
        List<Path> staged = new ArrayList<>(), published = new ArrayList<>();
        boolean complete = false;
        try {
            if (labels) {
                File target = stage.resolve(run.outputFile(source, "_labels.tif").getName()).toFile();
                staged.add(target.toPath());
                LabelMaskWriter.write(tensor, target);
            }
            if (rois) {
                SegmentationRegions regions = SegmentationRegions.fromLabels(tensor, kind, cancelled);
                File json = stage.resolve(run.outputFile(source, ".geojson").getName()).toFile();
                staged.add(json.toPath());
                SegmentationGeoJson.write(json, source, regions, cancelled);
                File xml = stage.resolve(run.outputFile(source, ".xml").getName()).toFile();
                staged.add(xml.toPath());
                if (host != null) host.saveSegmentationRois(source, xml, regions, cancelled);
                if (!xml.exists()) staged.remove(xml.toPath());
            }
            for (Path path : staged) {
                SegmentationRegions.check(cancelled);
                Path target = directory.resolve(path.getFileName());
                Files.move(path, target);
                published.add(target);
            }
            complete = true;
        } finally {
            if (!complete) for (Path path : published) Files.deleteIfExists(path);
            for (Path path : staged) Files.deleteIfExists(path);
            Files.deleteIfExists(stage);
        }
        for (Path path : published) log.accept("Saved " + path);
    }
}
