package io.bioimage.modelrunner.model.special.crossgoose;

import static org.junit.Assert.*;

import java.io.File;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import org.apposed.appose.Service.ResponseType;
import org.apposed.appose.TaskEvent;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.model.special.unet.UnetTrainingProgress;
import io.bioimage.modelrunner.model.special.unet.UnetValidationPreview;

public class CrossGooseBridgeTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test public void explicitCheckpointIsPreservedAndDirectoryDefaultsToBest() throws Exception {
        File model = temporary.newFolder("model");
        File checkpoints = new File(model, "checkpoints");
        assertTrue(checkpoints.mkdir());
        File best = new File(checkpoints, "best.ckpt"), last = new File(checkpoints, "last.ckpt");
        Files.write(best.toPath(), new byte[] {1});
        Files.write(last.toPath(), new byte[] {2});
        assertEquals(last, CrossGoose.modelPathForLoading(last));
        assertEquals(best, CrossGoose.modelPathForLoading(best));
        assertEquals(best, CrossGoose.modelPathForLoading(model));
    }

    @Test public void forwardsMetricsPreviewsAndCheckpointEventsWithoutChangingTheirMeaning() {
        Map<String, Object> info = new LinkedHashMap<>();
        info.put("type", "progress");
        info.put("epoch", 2); info.put("step", 8); info.put("total_epochs", 5); info.put("total_steps", 20);
        info.put("losses", Map.of("train/total_loss", 1.5, "val/total_loss", 1.2));
        info.put("metrics", Map.of("val/ap_0.5", .8, "val/ap_0.75", .6, "val/ap_0.9", .3));
        AtomicReference<UnetTrainingProgress> progress = new AtomicReference<>();
        AtomicReference<UnetValidationPreview> preview = new AtomicReference<>();
        List<String> log = new ArrayList<>();
        List<Map<String, Object>> validation = new ArrayList<>();
        CrossGoose.handleTrainingEvent(event(info, "Epoch completed"), progress::set, preview::set,
                log::add, validation::add);
        assertEquals(8, progress.get().getStep());
        assertEquals(1.5, progress.get().getTrainingTotalLoss(), 0);
        assertEquals(.6, progress.get().getPrimaryMetric(), 0);
        assertEquals("AP@0.75", progress.get().getPrimaryMetricName());
        info = new LinkedHashMap<>(Map.of("type", "preview", "epoch", 2,
                "preview_path", "epoch.json", "latest_preview_path", "latest.json"));
        CrossGoose.handleTrainingEvent(event(info, "Preview saved"), progress::set, preview::set,
                log::add, validation::add);
        assertEquals("epoch.json", preview.get().getPreviewJsonPath());
        info = new LinkedHashMap<>(Map.of("type", "checkpoint", "best", true, "checkpoint_path", "best.ckpt"));
        CrossGoose.handleTrainingEvent(event(info, "New best model"), progress::set, preview::set,
                log::add, validation::add);
        assertEquals(2, validation.size());
        assertEquals(Boolean.TRUE, validation.get(1).get("best"));
        assertEquals(Arrays.asList("Epoch completed", "Preview saved", "New best model"), log);
        CrossGoose.handleTrainingEvent(event(Map.of("type", "full_validation", "status", "ready",
                "supported", true), ""), null, null, null, validation::add);
        assertEquals("full_validation", validation.get(2).get("type"));
    }

    @Test public void generatedTrainingUsesCallbacksRatherThanBackendSpecificTaskSerialization() throws Exception {
        Method method = CrossGoose.class.getDeclaredMethod("buildTrainingCode", Map.class);
        method.setAccessible(true);
        String script = (String) method.invoke(null, Map.of("device", "cpu", "model_dir", "model"));
        assertTrue(script.contains("info=event"));
        assertTrue(script.contains("callback=_jdll_cross_goose_callback"));
        assertTrue(script.contains("cancellation=lambda: task.cancel_requested"));
        assertFalse(script.contains("jdll_cross_goose_train(_jdll_cross_goose_config, task=task)"));
        assertTrue(script.contains("task.outputs['model_dir']"));
        assertTrue(script.contains("control=_jdll_control"));
        assertTrue(script.contains("validation_control.file_control"));
    }

    private static TaskEvent event(Map<String, Object> info, String message) {
        return new TaskEvent(null, ResponseType.UPDATE, message, 0, 1, info);
    }
}
