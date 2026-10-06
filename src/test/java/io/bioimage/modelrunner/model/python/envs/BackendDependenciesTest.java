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
package io.bioimage.modelrunner.model.python.envs;

import static org.junit.Assert.*;

import java.io.File;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.model.special.denoise.Denoising;
import io.bioimage.modelrunner.model.special.unet.Unet;

public class BackendDependenciesTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test
    public void allPlatformsInheritTheSamePinnedBackendsAndExtras() {
        Map<String, String> revisions = new HashMap<>();
        for (String platform : new String[] {"lin-x86-cuda", "lin-x86-no-cuda",
                "win-x86-cuda", "win-x86-no-cuda", "mac-arm", "mac-x86"}) {
            String template = PixiEnvironmentResolver.readClasspathResourceAsString(
                    "tomls/biapy-pixi-" + platform + ".toml");
            String manifest = String.format(Locale.ROOT, template, "backend-test", "124", "124");
            Matcher table = Pattern.compile("(?ms)^\\[pypi-dependencies\\]\\R(.*?)(?=^\\[|\\z)")
                    .matcher(manifest);
            assertTrue(platform, table.find());
            for (String name : new String[] {"jdll-unet", "jdll-denoise"}) {
                String extra = name.equals("jdll-unet") ? "region-reading" : "bm";
                Matcher dependency = Pattern.compile("(?m)^" + name
                        + " = \\{ git = \"https://github\\.com/carlosuc3m/" + name
                        + "\\.git\", rev = \"([0-9a-f]{40})\", extras = \\[\"" + extra + "\"\\] \\}$")
                        .matcher(table.group(1));
                assertTrue(platform + ": missing pinned " + name + " with " + extra, dependency.find());
                String previous = revisions.putIfAbsent(name, dependency.group(1));
                if (previous != null) assertEquals(platform, previous, dependency.group(1));
                assertEquals("Do not override the shared dependency in platform features", 1,
                        manifest.split("(?m)^" + name + " =", -1).length - 1);
            }
            assertFalse("macOS variants must inherit the default feature", manifest.contains("no-default-feature"));
        }
    }

    @Test
    public void localSourcesAreExplicitOverridesOnly() throws Exception {
        Class<?>[] bridges = {Unet.class, Denoising.class};
        String[] properties = {"jdll.unet.path", "jdll.denoise.source"};
        String[] variables = {"JDLL_UNET_PATH", "JDLL_DENOISE_SOURCE"};
        String[] methods = {"jdllUnetSourceDir", "sourceDirectory"};
        String override = temporary.newFolder("explicit-development-source").getAbsolutePath();
        for (int i = 0; i < bridges.length; i++) {
            Method method = bridges[i].getDeclaredMethod(methods[i]);
            method.setAccessible(true);
            String previous = System.getProperty(properties[i]);
            try {
                System.clearProperty(properties[i]);
                String env = System.getenv(variables[i]);
                String expected = "";
                if (i == 0 && env != null) expected = env.trim();
                if (i == 1 && env != null && new File(env).isDirectory())
                    expected = new File(env).getAbsolutePath();
                assertEquals("No implicit local checkout", expected, method.invoke(null));
                System.setProperty(properties[i], override);
                assertEquals(override, method.invoke(null));
            } finally {
                if (previous == null) System.clearProperty(properties[i]);
                else System.setProperty(properties[i], previous);
            }
        }
    }
}
