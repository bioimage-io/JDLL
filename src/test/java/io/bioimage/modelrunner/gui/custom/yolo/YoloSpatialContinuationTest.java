package io.bioimage.modelrunner.gui.custom.yolo;

import static org.junit.Assert.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.imageio.ImageIO;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import io.bioimage.modelrunner.gui.custom.training.TrainingSplit;

public class YoloSpatialContinuationTest {
    @Rule public TemporaryFolder tmp = new TemporaryFolder();

    private void writeSample(Path data, String name, String labels) throws Exception {
        Path images = Files.createDirectories(data.resolve("train/images"));
        Path annotations = Files.createDirectories(data.resolve("train/labels"));
        ImageIO.write(new BufferedImage(100, 100, BufferedImage.TYPE_BYTE_GRAY), "png", images.resolve(name + ".png").toFile());
        Files.writeString(annotations.resolve(name + ".txt"), labels);
    }

    @Test public void preservesTheBackgroundRegionAndAddsAnAnnotatedTrainingImage() throws Exception {
        File data = tmp.newFolder("dataset"), models = tmp.newFolder("models");
        Path parent = tmp.newFolder("parent").toPath(), child = tmp.newFolder("child").toPath();
        String left = "0 0.15 0.5 0.10 0.10\n", right = "0 0.90 0.5 0.10 0.10\n";
        writeSample(data.toPath(), "old", left + right);
        YoloDatasetPreparer.prepareDetailed(data.toString(), "first", models.toString(), 128, null,
                new TrainingSplit(data, parent.toFile(), null, null));
        JsonObject original = JsonParser.parseString(Files.readString(parent.resolve("split.json"))).getAsJsonObject();
        Files.createFile(parent.resolve("best.pt"));
        Files.writeString(parent.resolve("continuation.json"), "{\"status\":\"cancelled\",\"epochs\":10,"
                + "\"checkpoints\":{\"best\":{\"weights\":\"best.pt\",\"epoch\":1}}}");
        writeSample(data.toPath(), "old", left);
        writeSample(data.toPath(), "new", right);
        YoloDatasetPreparer.prepareDetailed(data.toString(), "next", models.toString(), 128, null,
                new TrainingSplit(data, child.toFile(), parent.toString(), null));
        JsonObject saved = JsonParser.parseString(Files.readString(child.resolve("split.json"))).getAsJsonObject();
        assertEquals(original.get("spatial"), saved.get("spatial"));
        assertEquals(1, saved.getAsJsonArray("train").size());
        assertEquals(2, saved.getAsJsonArray("val").size());
        assertTrue(saved.getAsJsonArray("val").toString().contains("new.png"));
    }
}
