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
package io.bioimage.modelrunner.gui.custom.yolo;

import javax.swing.JCheckBox;
import javax.swing.SwingUtilities;
import java.util.function.Supplier;

import io.bioimage.modelrunner.model.python.DLModelPytorchProtected;
import io.bioimage.modelrunner.system.PlatformDetection;

final class YoloAccelerationCheckBox extends JCheckBox {

    private static final long serialVersionUID = 7698477740479444473L;

    YoloAccelerationCheckBox() {
        this(() -> DLModelPytorchProtected.resolvePytorchEnv()
                .getSelectedEnvironment()
                .toLowerCase()
                .contains("cuda"));
    }

    YoloAccelerationCheckBox(Supplier<Boolean> cudaAvailabilityCheck) {
        super(accelerationName() + " acceleration", false);
        setOpaque(true);
        setBackground(YoloUiUtils.PANEL_BG);
        setForeground(YoloUiUtils.SECONDARY_BUTTON_FG);
        setFocusPainted(false);
        setToolTipText("Enable hardware acceleration when available; falls back to CPU if unavailable.");
        configureInitialAvailability(cudaAvailabilityCheck);
    }

    private static String accelerationName() {
        return isAppleSilicon() ? "MPS" : "CUDA";
    }

    private void configureInitialAvailability(Supplier<Boolean> cudaAvailabilityCheck) {
        setSelected(false);
        if (isAppleSilicon()) {
            setEnabled(true);
            setVisible(true);
            return;
        }
        if (PlatformDetection.isMacOS()) {
            setEnabled(false);
            setVisible(false);
            return;
        }
        setEnabled(false);
        setVisible(true);
        Thread cudaCheck = new Thread(() -> {
            boolean cudaAvailable = false;
            try {
                cudaAvailable = Boolean.TRUE.equals(cudaAvailabilityCheck.get());
            } catch (Exception e) {
                cudaAvailable = false;
            }
            final boolean enabled = cudaAvailable;
            SwingUtilities.invokeLater(() -> {
                setEnabled(enabled);
                setSelected(false);
                setVisible(true);
                repaint();
            });
        }, "jdll-cuda-compatibility-check");
        cudaCheck.setDaemon(true);
        cudaCheck.start();
    }

    private static boolean isAppleSilicon() {
        return PlatformDetection.isMacOS()
                && (PlatformDetection.ARCH_ARM64.equals(PlatformDetection.getArch())
                || PlatformDetection.isUsingRosseta());
    }
}
