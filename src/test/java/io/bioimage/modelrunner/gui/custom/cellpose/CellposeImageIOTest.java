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

import java.awt.image.BufferedImage;
import java.io.File;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Collections;

import javax.imageio.ImageIO;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.gui.custom.CellposePluginUI;
import io.bioimage.modelrunner.gui.custom.InferenceRunFiles;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.real.FloatType;

public class CellposeImageIOTest {
    @Rule public TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void savesAllZTPlanesAsFloatLabelsAndDoesNotReprocessOrOverwriteThem() throws Exception {
        File source = new File(folder.newFolder("images"), "source.png");
        assertTrue(ImageIO.write(new BufferedImage(2, 2, BufferedImage.TYPE_BYTE_GRAY), "png", source));
        float[] values = {1, 70000, 90000, 120000, 130000, 140000};
        Tensor<FloatType> labels = Tensor.build("labels", "xyzt", ArrayImgs.floats(values, 1, 1, 2, 3));
        InferenceRunFiles run = InferenceRunFiles.create(source, "cellpose", "cyto3",
                Collections.singletonList(source), null);
        File output = run.outputFile(source, "_labels.tif");
        method("writeLabelMask", Tensor.class, File.class).invoke(null, labels, output);
        try (javax.imageio.stream.ImageInputStream input = ImageIO.createImageInputStream(output)) {
            javax.imageio.ImageReader reader = ImageIO.getImageReaders(input).next();
            try {
                reader.setInput(input);
                assertEquals(6, reader.getNumImages(true));
                for (int i = 0; i < values.length; i++)
                    assertEquals(values[i], reader.read(i).getRaster().getSampleFloat(0, 0, 0), 0);
            } finally { reader.dispose(); }
        }
        File next = InferenceRunFiles.create(source, "cellpose", "cyto3", Collections.singletonList(source), null)
                .outputFile(source, "_labels.tif");
        assertNotEquals(output, next);
        method("writeLabelMask", Tensor.class, File.class).invoke(null, labels, next);
        List<?> inputs = (List<?>) method("systemImages", File.class).invoke(null, source.getParentFile());
        assertEquals(1, inputs.size());
        assertEquals(source, inputs.get(0));
    }

    private static Method method(String name, Class<?>... parameters) throws Exception {
        Method method = CellposePluginUI.class.getDeclaredMethod(name, parameters);
        method.setAccessible(true);
        return method;
    }
}
