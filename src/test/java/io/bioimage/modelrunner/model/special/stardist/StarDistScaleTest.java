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
package io.bioimage.modelrunner.model.special.stardist;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class StarDistScaleTest {

    @Test
    public void metadataScaleMatchesEquivalentDiameterAndIsBounded() {
        double diameter = 2.0d * Math.sqrt(100.0d / Math.PI);

        assertEquals(1.0d, StarDist.metadataObjectScale(diameter, 10.0d, 10.0d, 1.0d, 1.0d), 1e-12);
        assertEquals(0.25d, StarDist.metadataObjectScale(1.0d, 100.0d, 100.0d, 1.0d, 1.0d), 1e-12);
        assertEquals(4.0d, StarDist.metadataObjectScale(100.0d, 1.0d, 1.0d, 1.0d, 1.0d), 1e-12);
    }

    @Test
    public void legacyVolumeScaleTargetsThreeDimensionalEquivalentOfAreaRule() {
        long tileX = 512L;
        long tileY = 512L;
        long tileZ = 64L;
        double tileVolume = tileX * (double) tileY * tileZ;
        double objectSide = Math.cbrt(0.008d * tileVolume);

        assertEquals(1.0d, StarDist.legacyVolumeObjectScale(
                objectSide, objectSide, tileX, tileY, tileZ, new double[] {1.0d, 1.0d, 1.0d}), 1e-12);

        double scale = StarDist.legacyVolumeObjectScale(
                1.0d, 1.0d, tileX, tileY, tileZ, new double[] {1.0d, 1.0d, 1.0d});
        assertEquals(0.008d, scale * scale * scale / tileVolume, 1e-12);
    }

    @Test
    public void legacyVolumeScaleUsesAnisotropyToEstimateDepth() {
        double isotropic = StarDist.legacyVolumeObjectScale(
                1.0d, 1.0d, 512L, 512L, 64L, new double[] {1.0d, 1.0d, 1.0d});
        double anisotropic = StarDist.legacyVolumeObjectScale(
                1.0d, 1.0d, 512L, 512L, 64L, new double[] {4.0d, 1.0d, 1.0d});

        assertEquals(Math.cbrt(4.0d), anisotropic / isotropic, 1e-12);
    }
}
