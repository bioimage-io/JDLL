package org.bioimageanalysis.icy.deeplearning.system;

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

    private String os;
    private String arch;
    private boolean rosetta = false;

    /**
     * Creates a platform detection using the local platform.
     */
    public PlatformDetection()
    {
    }

    /**
     * @return The operating system of the platform. e.g. windows, linux, macosx, etc.
     */
    public String getOs()
    {
    	if (os != null)
    		return os;
        // resolve OS
        if (System.getProperty("os.name").toLowerCase().contains("win"))
        {
            os = OS_WINDOWS;
        }
        else if (System.getProperty("os.name").toLowerCase().replace(" ", "").contains("macosx"))
        {
            os = OS_OSX;
        }
        else if (System.getProperty("os.name").toLowerCase().contains("linux") 
        		|| System.getProperty("os.name").toLowerCase().endsWith("ix"))
        {
            os = OS_LINUX;
        }
        else
        {
            throw new IllegalArgumentException("Operating system not supported by Miniconda: " + System.getProperty("os.name")
								+ ". Only supported OS are: " + OS_WINDOWS + ", " + OS_OSX + " and " + OS_LINUX);
        }
        return os;
    }

    /**
     * @return The system architecture. e.g. x86, x86_64, amd64, etc.
     */
    public String getArch()
    {
    	if (arch != null)
    		return arch;
        // resolve architecture
        arch = archMap.get(System.getProperty("os.arch"));
        if (this.arch == null)
        {
            throw new IllegalArgumentException("Unknown architecture " + System.getProperty("os.arch"));
        }
        if (this.arch.equals(ARCH_X86_64) && !getOs().equals(PlatformDetection.OS_WINDOWS)) {
			try {
				Process proc = Runtime.getRuntime().exec(
						new String[] {"bash", "-c", DETECT_CHIP_TERMINAL_COMMAND});
				String txt = waitProcessExecutionAndGetOutputText(proc);
	    		if (txt.toLowerCase().contains("apple m"))
	    			rosetta = true;
			} catch (IOException e) {
				e.printStackTrace();
				System.out.println("Error checking the chip architecture with bash");
			}
        }
        return arch;
    }
    
    public boolean isUsingRosseta() {
    	if (arch == null)
    		getArch();
    	return this.rosetta;
    }

    @Override
    public String toString()
    {
    	if (os == null)
    		getOs();
    	if (arch == null)
    		getArch();
        return os + "-" + arch;
    }
    
    public static boolean isWindows() {
    	return new PlatformDetection().getOs().equals(PlatformDetection.OS_WINDOWS);
    }
    
    public static boolean isLinux() {
    	return new PlatformDetection().getOs().equals(PlatformDetection.OS_LINUX);
    }
    
    public static boolean isMacOS() {
    	return new PlatformDetection().getOs().equals(PlatformDetection.OS_OSX);
    }
    
    public static String getPythonArchDetectionCommand() {
    	return PYTHON_ARCH_DETECTION_COMMAND;
    }
    
    public static String executeUnameM() {
    	if (UNAME_M != null)
    		return UNAME_M;
    	Process proc;
		try {
			proc = Runtime.getRuntime().exec(
					new String[] {"bash", "-c", "uname -m"});
			UNAME_M = archMap.get(waitProcessExecutionAndGetOutputText(proc));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return UNAME_M;
    }
	
	public static String waitProcessExecutionAndGetOutputText(Process proc) throws IOException {
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
	
	public static String readBufferedReaderIntoStringIntoString(BufferedReader input) throws IOException {
		String text = "";
		String line;
	    while ((line = input.readLine()) != null) {
	    	text += line + System.lineSeparator();
	    }
	    return text;
	}
}
