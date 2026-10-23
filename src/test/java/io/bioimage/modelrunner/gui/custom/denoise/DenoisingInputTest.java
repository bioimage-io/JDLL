package io.bioimage.modelrunner.gui.custom.denoise;

import static org.junit.Assert.*;

import org.junit.Test;

import io.bioimage.modelrunner.model.special.denoise.DenoisingConfig;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.real.FloatType;

public class DenoisingInputTest {
    @Test
    public void singletonZDoesNotTurnAPlanarImageIntoAVolumeForAnyMethod() {
        for (String method : new String[] {"noise2fast", "zs_n2n", "structn2v", "bm3d_bm4d"}) {
            DenoisingConfig config = DenoisingConfig.builder().method(method).axes("xyczt").seed(42).build();
            RandomAccessibleInterval<FloatType> image = ArrayImgs.floats(4, 3, 2, 1, 5);
            Tensor<FloatType> input = DenoisingService.prepareInput(config, image);
            assertEquals("xyct", input.getAxesOrderString());
            assertArrayEquals(new long[] {4, 3, 2, 5}, input.getData().dimensionsAsLongArray());
            DenoisingConfig prepared = config.withAxes(input.getAxesOrderString());
            assertEquals(method, prepared.getMethod());
            assertEquals(42, prepared.getSeed());
            RandomAccessibleInterval<FloatType> restored = DenoisingService.restoreAxes(
                    input.getData(), input.getAxesOrderString(), config.getAxes());
            assertArrayEquals(image.dimensionsAsLongArray(), restored.dimensionsAsLongArray());
            net.imglib2.RandomAccess<FloatType> access = restored.randomAccess();
            access.setPosition(new long[] {2, 1, 1, 0, 4});
            access.get().set(123);
            access = image.randomAccess();
            access.setPosition(new long[] {2, 1, 1, 0, 4});
            assertEquals(123, access.get().get(), 0);
        }
    }

    @Test
    public void realVolumesKeepTheirZAxis() {
        RandomAccessibleInterval<FloatType> image = ArrayImgs.floats(4, 3, 2, 6, 5);
        Tensor<FloatType> input = DenoisingService.prepareInput(
                DenoisingConfig.builder().axes("xyczt").build(), image);
        assertEquals("xyczt", input.getAxesOrderString());
        assertSame(image, input.getData());
        assertSame(image, DenoisingService.restoreAxes(image, "xyczt", "xyczt"));
    }
}
