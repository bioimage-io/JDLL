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
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.system.GpuCompatibility;

public final class YoloAccelerationCheckBox extends JCheckBox {

    public enum Backend { PYTORCH, TENSORFLOW }

    private static final long serialVersionUID = 7698477740479444473L;
    private boolean accelerationAvailable;
    private boolean enableRequested = true;
    private final String acceleratorName;

    public YoloAccelerationCheckBox() {
        this(DLModelPytorchProtected::resolvePytorchEnv, Backend.PYTORCH);
    }

    public YoloAccelerationCheckBox(Supplier<PixiEnvironmentSpec> environment, Backend backend) {
        this(() -> (!PlatformDetection.isMacOS() || isAppleSilicon())
                && AccelerationAvailability.check(environment.get(), backend), accelerationName());
    }

    public YoloAccelerationCheckBox(Supplier<Boolean> cudaAvailabilityCheck) {
        this(() -> isAppleSilicon() ? GpuCompatibility.isMetalAvailable()
                : !PlatformDetection.isMacOS() && GpuCompatibility.isCudaAvailable()
                        && Boolean.TRUE.equals(cudaAvailabilityCheck.get()), accelerationName());
    }

    // The supplied probe is complete, allowing deterministic checks without native hardware.
    YoloAccelerationCheckBox(Supplier<Boolean> availabilityCheck, String acceleratorName) {
        super(acceleratorName + " acceleration", false);
        this.acceleratorName = acceleratorName;
        setOpaque(true);
        setBackground(YoloUiUtils.PANEL_BG);
        setForeground(YoloUiUtils.SECONDARY_BUTTON_FG);
        setFocusPainted(false);
        setToolTipText("Checking hardware acceleration availability...");
        configureInitialAvailability(availabilityCheck);
    }

    @Override
    public void setEnabled(boolean enabled) {
        enableRequested = enabled;
        super.setEnabled(enabled && accelerationAvailable);
    }

    @Override
    public void setSelected(boolean selected) {
        super.setSelected(selected && accelerationAvailable);
    }

    public boolean isAccelerationAvailable() {
        return accelerationAvailable;
    }

    private static String accelerationName() {
        return isAppleSilicon() ? "MPS" : "CUDA";
    }

    private void configureInitialAvailability(Supplier<Boolean> availabilityCheck) {
        setSelected(false);
        super.setEnabled(false);
        setVisible(!PlatformDetection.isMacOS() || isAppleSilicon());
        Thread cudaCheck = new Thread(() -> {
            boolean cudaAvailable = false;
            try {
                cudaAvailable = Boolean.TRUE.equals(availabilityCheck.get());
            } catch (Exception | LinkageError e) {
                cudaAvailable = false;
            }
            final boolean enabled = cudaAvailable;
            SwingUtilities.invokeLater(() -> {
                accelerationAvailable = enabled;
                if (!enabled) setSelected(false);
                setToolTipText(enabled ? "Use " + acceleratorName + " acceleration."
                        : acceleratorName + " acceleration is unavailable: the GPU, driver or Python backend could not be used.");
                YoloAccelerationCheckBox.super.setEnabled(enableRequested && enabled);
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
