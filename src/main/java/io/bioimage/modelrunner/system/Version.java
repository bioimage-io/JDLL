/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.system;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Represents the detected platform in a given system. When a new instance is created it is assigned the local detected platform but it can be changed using the
 * available setters.
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public class Version
{
    private final int major;
    private final int minor;
    private final int patch;
    private final String preRelease;      // e.g. "beta", "RC1", "SNAPSHOT"
    private final String buildMetadata;   // e.g. "001", "exp.sha.5114f85"
	
	private static final Pattern VERSION_PATTERN = Pattern.compile(
	        "^" +
	        "(?<major>0|[1-9]\\d*)" +               // major
	        "(?:\\.(?<minor>0|[1-9]\\d*))?" +       // .minor (optional)
	        "(?:\\.(?<patch>0|[1-9]\\d*))?" +       // .patch (optional)
	        "(?:-(?<pre>[0-9A-Za-z-\\.]+))?" +      // -preRelease (optional)
	        "(?:\\+(?<build>[0-9A-Za-z-\\.]+))?" +  // +buildMetadata (optional)
	        "$"
	    );


    private Version(int major, int minor, int patch, String preRelease, String buildMetadata) {
        this.major = major;
        this.minor = minor;
        this.patch = patch;
        this.preRelease = preRelease;
        this.buildMetadata = buildMetadata;
    }

    public static Version parse(String version) {
        Matcher m = VERSION_PATTERN.matcher(version);
        if (!m.matches()) {
            throw new IllegalArgumentException("Invalid version string: " + version);
        }
        int maj = Integer.parseInt(m.group("major"));
        int min = m.group("minor") != null ? Integer.parseInt(m.group("minor")) : 0;
        int pat = m.group("patch") != null ? Integer.parseInt(m.group("patch")) : 0;
        String pre = m.group("pre");
        String build = m.group("build");
        return new Version(maj, min, pat, pre, build);
    }

    // Getters
    public int getMajor()       { return major; }
    public int getMinor()       { return minor; }
    public int getPatch()       { return patch; }
    public String getPreRelease()    { return preRelease; }
    public String getBuildMetadata() { return buildMetadata; }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(major).append('.').append(minor).append('.').append(patch);
        if (preRelease != null)    sb.append('-').append(preRelease);
        if (buildMetadata != null) sb.append('+').append(buildMetadata);
        return sb.toString();
    }

    // Example use
    public static void main(String[] args) {
        String[] examples = {
            "1.2.3",
            "2.0",
            "3",
            "4.5.6-RC1",
            "7.8-SNAPSHOT+exp.sha.5114f85",
            "10.0.1+001"
        };

        for (String ver : examples) {
            Version v = Version.parse(ver);
            System.out.println("Parsed “" + ver + "” → " +
                "major=" + v.getMajor() +
                ", minor=" + v.getMinor() +
                ", patch=" + v.getPatch() +
                ", preRelease=" + v.getPreRelease() +
                ", buildMetadata=" + v.getBuildMetadata()
            );
        }
    }
}
