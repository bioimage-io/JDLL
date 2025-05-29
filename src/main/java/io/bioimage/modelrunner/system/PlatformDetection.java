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

/**
 * Represents the detected platform in a given system. When a new instance is created it is assigned the local detected platform but it can be changed using the
 * available setters.
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public class PlatformDetection
{
    public static final String OS_WINDOWS = "windows";
    public static final String OS_OSX = "macosx";
    public static final String OS_LINUX = "linux";
    public static final String ARCH_PPC64LE = "ppc64le";
    public static final String ARCH_X86_32 = "x86";
    public static final String ARCH_X86_64 = "x86_64";
    public static final String ARCH_S390X = "s390x";
    // Aarch64 and arm64 are equivalent architectures
    public static final String ARCH_AARCH64 = "aarch64";
    public static final String ARCH_ARM64 = "arm64";
    public static final Map<String, String> archMap;

    static
    {
        Map<String, String> architectures = new HashMap<>();
        architectures.put("x86", ARCH_X86_32);
        architectures.put("i386", ARCH_X86_32);
        architectures.put("i486", ARCH_X86_32);
        architectures.put("i586", ARCH_X86_32);
        architectures.put("i686", ARCH_X86_32);
        architectures.put("x86_64", ARCH_X86_64);
        architectures.put("amd64", ARCH_X86_64);
        architectures.put("ppc64le", ARCH_PPC64LE);
        architectures.put("s390x", ARCH_S390X);
        architectures.put("arm64", ARCH_ARM64);
        architectures.put("aarch64", ARCH_ARM64);
        archMap = Collections.unmodifiableMap(architectures);
    }
    
    private static String DETECT_CHIP_TERMINAL_COMMAND = "sysctl -n machdep.cpu.brand_string";
    
    private static String PYTHON_ARCH_DETECTION_COMMAND = "\"import platform; print(platform.machine())\"";
    
    private static String UNAME_M;

    private static String OS;
    private static String ARCH;
    private static boolean ROSETTA = false;
    private static Integer JAVA_VERSION;

    /**
     * @return The operating system of the platform. e.g. windows, linux, macosx, etc.
     */
    public static String getOs()
    {
    	if (OS != null)
    		return OS;
        // resolve OS
        if (System.getProperty("os.name").toLowerCase().contains("win"))
        {
            OS = OS_WINDOWS;
        }
        else if (System.getProperty("os.name").toLowerCase().replace(" ", "").contains("macosx"))
        {
            OS = OS_OSX;
        }
        else if (System.getProperty("os.name").toLowerCase().contains("linux") 
        		|| System.getProperty("os.name").toLowerCase().endsWith("ix"))
        {
            OS = OS_LINUX;
        }
        else
        {
            throw new IllegalArgumentException("Operating system not supported by Miniconda: " + System.getProperty("os.name")
								+ ". Only supported OS are: " + OS_WINDOWS + ", " + OS_OSX + " and " + OS_LINUX);
        }
        return OS;
    }

    /**
     * @return The system architecture. e.g. x86, x86_64, amd64, etc.
     */
    public static String getArch()
    {
    	if (ARCH != null)
    		return ARCH;
        // resolve architecture
        ARCH = archMap.get(System.getProperty("os.arch"));
        if (ARCH == null)
        {
            throw new IllegalArgumentException("Unknown architecture " + System.getProperty("os.arch"));
        }
        if (ARCH.equals(ARCH_X86_64) && !getOs().equals(PlatformDetection.OS_WINDOWS)) {
			try {
				Process proc = Runtime.getRuntime().exec(
						new String[] {"bash", "-c", DETECT_CHIP_TERMINAL_COMMAND});
				String txt = waitProcessExecutionAndGetOutputText(proc);
	    		if (txt.toLowerCase().contains("apple m"))
	    			ROSETTA = true;
			} catch (IOException e) {
				e.printStackTrace();
				System.out.println("Error checking the chip architecture with bash");
			}
        }
        return ARCH;
    }
    
    public static boolean isUsingRosseta() {
    	if (ARCH == null)
    		getArch();
    	return ROSETTA;
    }

    @Override
    public String toString()
    {
    	if (OS == null)
    		getOs();
    	if (ARCH == null)
    		getArch();
        return OS + "-" + ARCH;
    }
    
    public static boolean isWindows() {
    	return getOs().equals(PlatformDetection.OS_WINDOWS);
    }
    
    public static boolean isLinux() {
    	return getOs().equals(PlatformDetection.OS_LINUX);
    }
    
    public static boolean isMacOS() {
    	return getOs().equals(PlatformDetection.OS_OSX);
    }
	
    private static String waitProcessExecutionAndGetOutputText(Process proc) throws IOException {
		BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(proc.getInputStream()));
		String inputStreamTxt = "";
		boolean first = true;
		while (proc.isAlive() || first) {
			first = false;
			try {
				String txtAux1 = readBufferedReaderIntoStringIntoString(bufferedReader);
				inputStreamTxt += txtAux1;
				Thread.sleep(10);
			} catch (InterruptedException e) {
				throw new IOException("Interrumped process");
			}
		}
		bufferedReader.close();
		return inputStreamTxt;
	}
	
	private static String readBufferedReaderIntoStringIntoString(BufferedReader input) throws IOException {
		String text = "";
		String line;
	    while ((line = input.readLine()) != null) {
	    	text += line + System.lineSeparator();
	    }
	    return text;
	}
	
	public static Version getOSVersion() {
		return Version.parse(System.getProperty("os.version"));
	}
	
	/**
	 * Get the major Java version the program is currently running on
	 * @return the major Java version the program is running on
	 */
	public static int getJavaVersion() {
		if (JAVA_VERSION != null)
			return JAVA_VERSION;
		String version = System.getProperty("java.version");
	    if(version.startsWith("1.")) {
	        version = version.substring(2, 3);
	    } else {
	        int dot = version.indexOf(".");
	        if(dot != -1) { version = version.substring(0, dot); }
	    } 
	    JAVA_VERSION = Integer.parseInt(version);
	    return JAVA_VERSION;
	}
	
	public static void main(String[] args) {
		System.out.println(System.getProperty("os.version"));
	}
}
