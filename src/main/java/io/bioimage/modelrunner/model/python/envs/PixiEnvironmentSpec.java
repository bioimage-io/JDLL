package io.bioimage.modelrunner.model.python.envs;

import java.io.File;
import java.util.Objects;

/**
 * Immutable description of the Python environment that should be used for
 * PyTorch model execution on the current machine.
 */
public final class PixiEnvironmentSpec {

    private final String selectedEnvironment;
    private final String pixiTomlContent;
    private final File environmentDirectory;
    private final boolean installBiapyNoDeps;

    /**
     * Creates a new immutable environment specification.
     *
     * @param selectedEnvironment
     *     the Pixi environment name to use
     * @param pixiTomlContent
     *     the fully rendered pixi.toml content used to build the environment
     * @param environmentDirectory
     *     the directory where the environment is expected to live
     * @param installBiapyNoDeps
     *     whether a follow-up {@code pip install biapy==3.5.10 --no-deps}
     *     step is required after Pixi installs the environment
     */
    public PixiEnvironmentSpec(
            final String selectedEnvironment,
            final String pixiTomlContent,
            final File environmentDirectory,
            final boolean installBiapyNoDeps) {
        this.selectedEnvironment = Objects.requireNonNull(selectedEnvironment, "selectedEnvironment");
        this.pixiTomlContent = Objects.requireNonNull(pixiTomlContent, "pixiTomlContent");
        this.environmentDirectory = Objects.requireNonNull(environmentDirectory, "environmentDirectory");
        this.installBiapyNoDeps = installBiapyNoDeps;
    }

    /**
     * Creates a new immutable environment specification.
     *
     * @param selectedEnvironment
     *     the Pixi environment name to use
     * @param pixiTomlContent
     *     the fully rendered pixi.toml content used to build the environment
     * @param environmentDirectory
     *     the directory where the environment is expected to live
     */
    public PixiEnvironmentSpec(
            final String selectedEnvironment,
            final String pixiTomlContent,
            final File environmentDirectory) {
        this.selectedEnvironment = Objects.requireNonNull(selectedEnvironment, "selectedEnvironment");
        this.pixiTomlContent = Objects.requireNonNull(pixiTomlContent, "pixiTomlContent");
        this.environmentDirectory = Objects.requireNonNull(environmentDirectory, "environmentDirectory");
        this.installBiapyNoDeps = false;
    }

    /**
     * @return the selected Pixi environment name
     */
    public String getSelectedEnvironment() {
        return selectedEnvironment;
    }

    /**
     * @return the rendered pixi.toml content used to build the environment
     */
    public String getPixiTomlContent() {
        return pixiTomlContent;
    }

    /**
     * @return the directory where the environment is expected to live
     */
    public File getEnvironmentDirectory() {
        return environmentDirectory;
    }

    /**
     * @return whether Biapy must be installed afterwards with {@code --no-deps}
     */
    public boolean isInstallBiapyNoDeps() {
        return installBiapyNoDeps;
    }
}