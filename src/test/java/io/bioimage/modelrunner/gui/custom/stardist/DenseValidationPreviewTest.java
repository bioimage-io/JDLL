/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.gui.custom.stardist;

import static org.junit.Assert.*;
import java.lang.reflect.Method;
import org.junit.Test;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.integer.UnsignedIntType;

public class DenseValidationPreviewTest {
    @Test
    public void volumetricChannelsAreMovedWithoutConvertingLabelPrecision() throws Exception {
        RandomAccessibleInterval<?> input = ArrayImgs.unsignedInts(3, 4, 5, 6);
        RandomAccessibleInterval<?> output = channelsLast(input, 1);
        assertArrayEquals(new long[] {4, 5, 6, 3}, output.dimensionsAsLongArray());
        assertTrue(output.randomAccess().get() instanceof UnsignedIntType);
    }

    @Test
    public void fast3dPreviewShowsOnlyTheCentralContextPlanePerChannel() throws Exception {
        int[] values = new int[10 * 2 * 3];
        for (int i = 0; i < values.length; i++) values[i] = 70000 + i % 10;
        RandomAccessibleInterval<?> output = channelsLast(ArrayImgs.unsignedInts(values, 10, 2, 3), 5);
        assertArrayEquals(new long[] {2, 3, 2}, output.dimensionsAsLongArray());
        net.imglib2.RandomAccess<?> access = output.randomAccess();
        access.setPosition(new long[] {0, 0, 0});
        assertEquals(70002, ((UnsignedIntType) access.get()).getIntegerLong());
        access.setPosition(1, 2);
        assertEquals(70007, ((UnsignedIntType) access.get()).getIntegerLong());
    }

    private static RandomAccessibleInterval<?> channelsLast(RandomAccessibleInterval<?> input, int context) throws Exception {
        Method method = StardistValidationPreviewPanel.class.getDeclaredMethod("channelsLast",
                RandomAccessibleInterval.class, boolean.class, int.class);
        method.setAccessible(true);
        return (RandomAccessibleInterval<?>) method.invoke(null, input, true, context);
    }
}
