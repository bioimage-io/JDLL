package io.bioimage.modelrunner.model.python.envs;

import static org.junit.Assert.*;

import org.junit.Test;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;
import java.lang.reflect.Constructor;
import java.util.function.Consumer;

import io.bioimage.modelrunner.model.python.DLModelPytorchProtected;
import io.bioimage.modelrunner.model.special.crossgoose.CrossGoose;

public class PixiEnvironmentResolverTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test public void crossGooseAcceptsModelDirectoriesAndUsesTheMainEnvironment() throws Exception {
        java.io.File directory = temporary.newFolder("cross-goose-model");
        assertTrue(new java.io.File(directory, "weights.ckpt").createNewFile());
        Constructor<CrossGoose> constructor = CrossGoose.class.getDeclaredConstructor(String.class, Consumer.class, String.class);
        constructor.setAccessible(true);
        CrossGoose model = constructor.newInstance(directory.getAbsolutePath(), null, "cpu");
        assertEquals(CrossGoose.resolvePytorchEnv().getEnvironmentDirectory().getAbsolutePath(), model.getEnvPath());
        PixiEnvironmentSpec main = DLModelPytorchProtected.resolvePytorchEnv();
        assertEquals(main.getEnvironmentDirectory(), CrossGoose.resolvePytorchEnv().getEnvironmentDirectory());
        assertEquals(main.getPixiTomlContent(), CrossGoose.resolvePytorchEnv().getPixiTomlContent());
    }
    @Test public void platformsAndCudaBuildsResolveWithoutHardwareDetection() {
        for (String platform : new String[] {"linux-64", "win-64", "osx-64", "osx-arm64"}) {
            PixiEnvironmentSpec spec = PixiEnvironmentResolver.pytorch(platform, null, "default");
            assertEquals("default", spec.getSelectedEnvironment());
            assertTrue(spec.getEnvironmentDirectory().getName().startsWith("jdll-"));
            assertTrue(spec.getPixiTomlContent().contains("platforms = [\"" + platform + "\"]"));
            assertFalse(spec.getPixiTomlContent().contains("%s"));
            assertTrue(spec.getManualNoDeps().isEmpty());
            assertTrue(spec.getPixiTomlContent().contains("jdll-cross-goose ="));
            assertFalse(spec.getEnvironmentDirectory().getName().contains("cross-goose"));
        }
        for (String platform : new String[] {"linux-64", "win-64"}) {
            for (String cuda : new String[] {"11.8", "12.1", "12.4"}) {
                PixiEnvironmentSpec spec = PixiEnvironmentResolver.pytorch(platform, cuda, "default");
                assertTrue(spec.getEnvironmentDirectory().getName().endsWith(cuda.replace(".", "")));
                assertTrue(spec.getPixiTomlContent().contains("/whl/cu" + cuda.replace(".", "")));
            }
        }
    }

    @Test public void oldIntelProfilesApplyToEveryBackend() {
        for (String profile : new String[] {"macos-big-sur", "macos-catalina"}) {
            PixiEnvironmentSpec spec = PixiEnvironmentResolver.pytorch("osx-64", null, profile);
            assertEquals("default", spec.getSelectedEnvironment());
            assertTrue(spec.getEnvironmentDirectory().getName().endsWith(profile));
            assertTrue(spec.getPixiTomlContent().contains("cpu_generic*"));
            assertTrue(spec.getPixiTomlContent().contains("jdll-cross-goose ="));
            assertTrue(spec.getPixiTomlContent().contains("default = [\"runtime\", \""
                    + profile.replace("macos-", "pytorch-") + "\", \"core\"]"));
        }
    }

    @Test public void opencvCompatibilityResourcesAreExtractedAndChangesAreDetected() throws Exception {
        PixiEnvironmentSpec original = PixiEnvironmentResolver.pytorch("linux-64", null, "default");
        java.io.File directory = temporary.newFolder("main");
        PixiEnvironmentSpec spec = new PixiEnvironmentSpec("default", original.getPixiTomlContent(), directory);
        assertFalse(PixiEnvironmentManager.resourcesMatch(spec));
        PixiEnvironmentManager.prepareResources(spec);
        assertTrue(PixiEnvironmentManager.resourcesMatch(spec));
        java.io.File project = new java.io.File(directory, "opencv-python-compat/pyproject.toml");
        java.nio.file.Files.write(project.toPath(), new byte[0]);
        assertFalse(PixiEnvironmentManager.resourcesMatch(spec));
        PixiEnvironmentManager.prepareResources(spec);
        assertTrue(PixiEnvironmentManager.resourcesMatch(spec));
    }

    @Test public void legacyIsExplicitAndStillUsesTheOriginalManifests() {
        String previous = System.getProperty(PixiEnvironmentResolver.FAMILY_PROPERTY);
        try {
            System.setProperty(PixiEnvironmentResolver.FAMILY_PROPERTY, "legacy");
            assertTrue(PixiEnvironmentResolver.useLegacy());
            assertTrue(DLModelPytorchProtected.resolvePytorchEnv().getPixiTomlContent().contains("biapy ="));
            System.setProperty(PixiEnvironmentResolver.FAMILY_PROPERTY, "slim");
            assertFalse(PixiEnvironmentResolver.useLegacy());
            assertFalse(DLModelPytorchProtected.resolvePytorchEnv().getPixiTomlContent().contains("biapy ="));
            assertEquals(DLModelPytorchProtected.resolvePytorchEnv().getEnvironmentDirectory(),
                    CrossGoose.resolvePytorchEnv().getEnvironmentDirectory());
        } finally {
            if (previous == null) System.clearProperty(PixiEnvironmentResolver.FAMILY_PROPERTY);
            else System.setProperty(PixiEnvironmentResolver.FAMILY_PROPERTY, previous);
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void unsupportedCudaIsNotSilentlyMappedToAnotherBuild() {
        PixiEnvironmentResolver.pytorch("linux-64", "12.7", "default");
    }

    @Test(expected = IllegalArgumentException.class)
    public void unknownEnvironmentIsRejected() {
        PixiEnvironmentResolver.pytorch("linux-64", null, "typo");
    }
}
