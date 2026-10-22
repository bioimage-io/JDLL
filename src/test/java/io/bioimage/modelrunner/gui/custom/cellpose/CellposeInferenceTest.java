/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * #L%
 */
package io.bioimage.modelrunner.gui.custom.cellpose;

import static org.junit.Assert.*;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;

import javax.swing.JComboBox;
import javax.swing.SwingUtilities;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.model.special.cellpose.Cellpose;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

public class CellposeInferenceTest {
    @Rule public TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void packsArbitrarySourceChannelsWithoutRescalingIntensity() {
        RandomAccessibleInterval<FloatType> source = ArrayImgs.floats(new float[] {10, 20, 30, 40, 50}, 1, 1, 5);
        assertArrayEquals(new float[] {50, 20, 0}, values(CellposeChannels.pack(source, new int[] {5, 2})), 0);
        assertArrayEquals(new float[] {30, 0, 0}, values(CellposeChannels.pack(source, new int[] {0, 0})), 0);
        assertArrayEquals(new float[] {30, 40, 0}, values(CellposeChannels.pack(source, new int[] {0, 4})), 0);
        assertThrows(IllegalArgumentException.class, () -> CellposeChannels.pack(source, new int[] {6, 0}));
    }

    @Test
    public void selectorsPreserveValidSelectionsAndRejectUnavailableChannels() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            JComboBox<String> cyto = new JComboBox<>(), nuclei = new JComboBox<>();
            CellposeChannels.configure(cyto, nuclei, 5);
            cyto.setSelectedItem("channel 5");
            nuclei.setSelectedItem("channel 2");
            CellposeChannels.configure(cyto, nuclei, 5);
            assertEquals(5, CellposeChannels.parse((String) cyto.getSelectedItem()));
            assertEquals(2, CellposeChannels.parse((String) nuclei.getSelectedItem()));
            CellposeChannels.configure(cyto, nuclei, 1);
            assertEquals("gray", cyto.getSelectedItem());
            assertEquals("none", nuclei.getSelectedItem());
        });
    }

    @Test
    public void preservesAllPlanesAndEstimatesDiameterIndependentlyWithoutCarryOver() throws Exception {
        FakeModel model = new FakeModel(folder.newFile("weights"));
        CellposeInferenceService service = service(model);
        CellposeInferenceService.Run run = new CellposeInferenceService.Run("custom", new int[] {1, 0}, null, "cpu", false);
        RandomAccessibleInterval<FloatType> input = ArrayImgs.floats(2, 3, 1, 4, 5);
        for (int t = 0; t < 5; t++) for (int z = 0; z < 4; z++)
            for (FloatType value : Views.iterable(Views.hyperSlice(Views.hyperSlice(input, 4, t), 3, z)))
                value.set(t * 4 + z);
        List<Tensor<FloatType>> outputs = service.run(run, input, null);
        assertEquals(1, outputs.size());
        assertEquals("xyzt", outputs.get(0).getAxesOrderString());
        assertArrayEquals(new long[] {2, 3, 4, 5}, outputs.get(0).getData().dimensionsAsLongArray());
        assertEquals(20, model.primaryPixels.size());
        for (int i = 0; i < 20; i++) assertEquals((float) i, model.primaryPixels.get(i), 0);
        assertEquals(70000, outputs.get(0).getData().randomAccess().get().get(), 0);
        for (Float diameter : model.diameters) assertNull(diameter);
        service.run(run, ArrayImgs.floats(2, 3), null);
        assertNull(model.diameters.get(20));
        service.run(new CellposeInferenceService.Run("custom", new int[] {0, 0}, null, "cpu", false),
                ArrayImgs.floats(2, 3), null);
        assertNull(model.diameters.get(21));
        service.close();
    }

    @Test
    public void manualDiameterAndOptionalIntermediateShapesAreStable() throws Exception {
        FakeModel model = new FakeModel(folder.newFile("weights"));
        CellposeInferenceService service = service(model);
        CellposeInferenceService.Run run = new CellposeInferenceService.Run("custom", new int[] {3, 2}, 17f, "cpu", true);
        List<Tensor<FloatType>> outputs = service.run(run, ArrayImgs.floats(2, 3, 3, 2, 1), null);
        assertEquals(5, outputs.size());
        assertEquals(Arrays.asList(17f, 17f), model.diameters);
        assertArrayEquals(new long[] {2, 3, 2, 2, 1}, outputs.get(4).getData().dimensionsAsLongArray());
        service.run(run, ArrayImgs.floats(2, 3, 3), null);
        assertEquals(17f, model.diameters.get(2), 0);
        service.run(new CellposeInferenceService.Run("custom", new int[] {0, 0}, null, "cpu", false),
                ArrayImgs.floats(2, 3), null);
        assertNull(model.diameters.get(3));
        service.close();
    }

    @Test
    public void generatedTasksAlwaysIncludeCurrentDiameterAndOnlyRequestedOutputs() throws Exception {
        FakeModel model = new FakeModel(folder.newFile("model's weights"));
        model.setDiameter(42);
        assertTrue(model.inputCode().startsWith("diameter = 42.0"));
        assertTrue(model.inputCode().startsWith("diameter = 42.0"));
        model.clearDiameter();
        model.setIncludeIntermediateOutputs(false);
        String code = model.inputCode();
        assertTrue(code.startsWith("diameter = None"));
        assertTrue(code.contains(" = [_jdll_first(_jdll_masks)]"));
        assertTrue(code.contains("channel_axis=-1"));
        model.setIncludeIntermediateOutputs(true);
        assertTrue(model.inputCode().contains("_jdll_flows0[2]"));
        assertTrue(model.loadCode().contains("def resolve_cellpose_diameter("));
        assertTrue(model.loadCode().contains("pretrained_model=\""));
    }

    @Test
    public void runCopiesChannelSettingsBeforeBackgroundProcessing() throws Exception {
        FakeModel model = new FakeModel(folder.newFile("weights"));
        CellposeInferenceService service = service(model);
        int[] channels = {1, 0};
        CellposeInferenceService.Run run = new CellposeInferenceService.Run("custom", channels, 20f, "cpu", false);
        channels[0] = 2;
        service.run(run, ArrayImgs.floats(new float[] {10, 20}, 1, 1, 2), null);
        assertEquals(10f, model.primaryPixels.get(0), 0);
        service.close();
    }

    private CellposeInferenceService service(FakeModel model) {
        CellposeInstaller installer = new CellposeInstaller(null) {
            @Override public String installAndResolve(String selection, Consumer<String> log) { return "custom"; }
        };
        return new CellposeInferenceService(installer) {
            @Override protected Cellpose createModel(String path, String device) { return model; }
        };
    }

    private static float[] values(RandomAccessibleInterval<FloatType> image) {
        float[] values = new float[(int) Views.iterable(image).size()];
        int i = 0;
        for (FloatType value : Views.flatIterable(image)) values[i++] = value.get();
        return values;
    }

    private static final class FakeModel extends Cellpose {
        final List<Float> diameters = new ArrayList<>();
        final List<Float> primaryPixels = new ArrayList<>();
        boolean intermediates;

        FakeModel(File weights) throws Exception { super(null, null, weights.getAbsolutePath(), null, null, "cpu"); }
        @Override public void loadModel() { loaded = true; }
        @Override public Float getResolvedDiameter() { return 23f; }
        @Override public void setIncludeIntermediateOutputs(boolean include) {
            super.setIncludeIntermediateOutputs(include);
            intermediates = include;
        }
        String inputCode() { return createInputsCode(Collections.emptyList(), Collections.emptyList()); }
        String loadCode() throws java.io.IOException { return buildModelCode(); }

        @Override @SuppressWarnings({"rawtypes", "unchecked"})
        protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
        List<Tensor<R>> backboneSingleInferenceTile(List<Tensor<T>> inputs) {
            // Use the real public inference path and merger, without a Python model.
            diameters.add(getDiameter());
            RandomAccessibleInterval<T> data = inputs.get(0).getData();
            primaryPixels.add(data.randomAccess().get().getRealFloat());
            long x = data.dimension(0), y = data.dimension(1);
            RandomAccessibleInterval<UnsignedIntType> labels = ArrayImgs.unsignedInts(x, y);
            for (UnsignedIntType label : Views.iterable(labels)) label.set(70000);
            List<Tensor<?>> outputs = new ArrayList<>();
            outputs.add(Tensor.build("labels", "xy", labels));
            if (intermediates) {
                outputs.add(Tensor.build("rgb", "xyc", ArrayImgs.floats(x, y, 3)));
                outputs.add(Tensor.build("flow", "cxy", ArrayImgs.floats(2, x, y)));
                outputs.add(Tensor.build("prob", "xy", ArrayImgs.floats(x, y)));
                outputs.add(Tensor.build("styles", "i", ArrayImgs.floats(256)));
                outputs.add(Tensor.build("restored", "xyc", ArrayImgs.floats(x, y, diameters.size() % 2 + 1)));
            }
            return (List) outputs;
        }
    }
}
