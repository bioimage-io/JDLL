package org.bioimageanalysis.icy.deeplearning.system;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang.SystemUtils;

/**
 * Represents the detected platform in a given system. When a new instance is created it is assigned the local detected platform but it can be changed using the
 * available setters.
 *
 * @author Daniel Felipe Gonzalez Obando
 */
public class PlatformDetection {
    public static final String OS_WINDOWS = "windows";
    public static final String OS_OSX = "macosx";
    public static final String OS_SOLARIS = "solaris";
    public static final String OS_LINUX = "linux";
    public static final String ARCH_PPC = "ppc";
    public static final String ARCH_X86_32 = "x86";
    public static final String ARCH_X86_64 = "x86_64";
    public static final Map<String, String> archMap;

    static {
        Map<String, String> architectures = new HashMap<>();
        architectures.put("x86", ARCH_X86_32);
        architectures.put("i386", ARCH_X86_32);
        architectures.put("i486", ARCH_X86_32);
        architectures.put("i586", ARCH_X86_32);
        architectures.put("i686", ARCH_X86_32);
        architectures.put("x86_64", ARCH_X86_64);
        architectures.put("amd64", ARCH_X86_64);
        architectures.put("powerpc", ARCH_PPC);
        archMap = Collections.unmodifiableMap(architectures);
    }

    private String os;
    private String arch;

    /**
     * Creates a platform detection using the local platform.
     */
    public PlatformDetection() {
        // resolve OS
        if (SystemUtils.IS_OS_WINDOWS) {
            this.setOs(OS_WINDOWS);
        } else if (SystemUtils.IS_OS_MAC_OSX) {
            this.setOs(OS_OSX);
        } else if (SystemUtils.IS_OS_SOLARIS) {
            this.setOs(OS_SOLARIS);
        } else if (SystemUtils.IS_OS_LINUX) {
            this.setOs(OS_LINUX);
        } else {
            throw new IllegalArgumentException("Unknown operating system " + SystemUtils.OS_NAME);
        }

        // resolve architecture
        this.setArch(archMap.get(SystemUtils.OS_ARCH));
        if (this.arch == null) {
            throw new IllegalArgumentException("Unknown architecture " + SystemUtils.OS_ARCH);
        }
    }

    /**
     * @return The operating system of the platform. e.g. windows, linux, macosx, etc.
     */
    public String getOs() {
        return os;
    }

    /**
     * The Operating System of this detection.
     *
     * @param os The operating system to be assigned.
     */
    public void setOs(String os) {
        this.os = os;
    }

    /**
     * @return The system architecture. e.g. x86, x86_64, amd64, etc.
     */
    public String getArch() {
        return arch;
    }

    /**
     * Sets the system architecture of this detection.
     *
     * @param arch The system architecture to be assigned.
     */
    public void setArch(String arch) {
        this.arch = arch;
    }

    @Override
    public String toString() {

        return os + "-" + arch;
    }
}
