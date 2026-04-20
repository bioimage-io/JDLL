package io.bioimage.modelrunner.model.python.envs;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Immutable description of the Python environment that should be used for
 * PyTorch model execution on the current machine.
 */
public final class PixiEnvironmentSpec {

    private final String selectedEnvironment;
    private final String pixiTomlContent;
    private final File environmentDirectory;
    private final List<String> manualNoDeps;

    /**
     * Creates a new immutable environment specification.
     *
     * @param selectedEnvironment
     *     the Pixi environment name to use
     * @param pixiTomlContent
     *     the fully rendered pixi.toml content used to build the environment
     * @param environmentDirectory
     *     the directory where the environment is expected to live
     * @param manualNoDeps
     *     a list of deps installed with no deps. For example{@code pip install biapy==3.5.10 --no-deps}
     */
    public PixiEnvironmentSpec(
            final String selectedEnvironment,
            final String pixiTomlContent,
            final File environmentDirectory,
            final List<String> manualNoDeps) {
        this.selectedEnvironment = Objects.requireNonNull(selectedEnvironment, "selectedEnvironment");
        this.pixiTomlContent = Objects.requireNonNull(pixiTomlContent, "pixiTomlContent");
        this.environmentDirectory = Objects.requireNonNull(environmentDirectory, "environmentDirectory");
        if (manualNoDeps == null)
        	this.manualNoDeps = new ArrayList<String>();
        else
        	this.manualNoDeps = manualNoDeps;
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
        this.manualNoDeps = new ArrayList<String>();
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
     * @return a list of deps that will be installed with {@code --no-deps}
     */
    public List<String> getManualNoDeps() {
        return manualNoDeps;
    }
}