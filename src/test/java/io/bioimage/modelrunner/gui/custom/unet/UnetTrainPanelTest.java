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
package io.bioimage.modelrunner.gui.custom.unet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.LinkedHashMap;

import javax.swing.SwingUtilities;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.gui.custom.unet.UnetDatasetInspector.Dimensionality;

public class UnetTrainPanelTest {
    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void offersExpectedArchitecturesAndDefaultsOnTheEdt() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            UnetTrainPanel panel = new UnetTrainPanel();
            assertEquals(4, panel.getScratchArchitectureComboBox().getItemCount());
            assertEquals(UnetModelRegistry.SMALL_2D, panel.getSelectedScratchArchitectureValue());
            panel.setDatasetDimensionality(Dimensionality.TWO_D);
            assertEquals(4, panel.getScratchArchitectureComboBox().getItemCount());
            panel.setDatasetDimensionality(Dimensionality.THREE_D);
            assertEquals(8, panel.getScratchArchitectureComboBox().getItemCount());
            assertEquals(UnetModelRegistry.SMALL_FAST_3D, panel.getSelectedScratchArchitectureValue());
            assertFalse(panel.selectScratchArchitectureValue(UnetModelRegistry.SMALL_2D));
            panel.setDatasetDimensionality(Dimensionality.MIXED);
            assertEquals(12, panel.getScratchArchitectureComboBox().getItemCount());
            assertTrue(panel.selectScratchArchitectureValue(UnetModelRegistry.SMALL_2D));
            panel.setDatasetDimensionality(Dimensionality.UNKNOWN);
            assertEquals(4, panel.getScratchArchitectureComboBox().getItemCount());
        });
    }

    @Test
    public void defaultsToCompatibleCustomConfigsOnlyAfterReview() throws Exception {
        for (String architecture : new String[] {UnetModelRegistry.SMALL_2D, UnetModelRegistry.SMALL_TRUE_3D}) {
            File models = temporaryFolder.newFolder(architecture);
            File configFile = new File(models, "unet/custom/config.json");
            Files.createDirectories(configFile.toPath().getParent());
            Files.write(configFile.toPath(), ("{\"architecture\":\"" + architecture + "\"}")
                    .getBytes(StandardCharsets.UTF_8));
            SwingUtilities.invokeAndWait(() -> {
                UnetTrainPanel panel = new UnetTrainPanel();
                panel.setModelsDir(models.getAbsolutePath());
                panel.getModelNameField().setText("custom");
                panel.refreshScratchArchitectures();
                assertEquals(4, panel.getScratchArchitectureComboBox().getItemCount());
                panel.setDatasetDimensionality(Dimensionality.MIXED);
                assertEquals(13, panel.getScratchArchitectureComboBox().getItemCount());
                assertEquals(configFile.getAbsolutePath(), panel.getSelectedScratchArchitectureValue());
                panel.setDatasetDimensionality(architecture.endsWith("3d")
                        ? Dimensionality.TWO_D : Dimensionality.THREE_D);
                assertFalse(panel.selectScratchArchitectureValue(configFile.getAbsolutePath()));
            });
        }
    }

    @Test
    public void compatibilityPreservesSourceDimensionality() {
        for (Dimensionality dimensions : Dimensionality.values()) {
            assertEquals(dimensions != Dimensionality.THREE_D,
                    UnetModelRegistry.isCompatibleWithDataset(UnetModelRegistry.SMALL_2D, dimensions));
            assertEquals(dimensions != Dimensionality.TWO_D,
                    UnetModelRegistry.isCompatibleWithDataset(UnetModelRegistry.SMALL_FAST_3D, dimensions));
            assertEquals(dimensions != Dimensionality.TWO_D,
                    UnetModelRegistry.isCompatibleWithDataset(UnetModelRegistry.SMALL_TRUE_3D, dimensions));
        }
    }

    @Test
    public void sharedCrossGoosePanelKeepsItsFixedArchitectures() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            UnetTrainPanel panel = new UnetTrainPanel();
            LinkedHashMap<String, String> architectures = new LinkedHashMap<String, String>();
            architectures.put("Default", "default");
            panel.configureFixedArchitectures(architectures, "default", path -> true,
                    value -> true, "Weights", "pt");
            panel.setDatasetDimensionality(Dimensionality.MIXED);
            assertEquals(1, panel.getScratchArchitectureComboBox().getItemCount());
            assertEquals("default", panel.getSelectedScratchArchitectureValue());
        });
    }
}
