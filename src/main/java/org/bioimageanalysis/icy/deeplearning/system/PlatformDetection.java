package org.bioimageanalysis.icy.deeplearning.system;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;


/**
 * Represents the detected platform in a given system. When a new instance is
 * created it is assigned the local detected platform but it can be changed
 * using the available setters.
 * 
 * @author Carlos Javier Garcia Lopez de Haro
 * @author Daniel Felipe Gonzalez Obando
 */
public class PlatformDetection
{
	public static final String OS_WINDOWS = "windows";

	public static final String OS_OSX = "macosx";

	public static final String OS_SOLARIS = "solaris";

	public static final String OS_LINUX = "linux";

	public static final String ARCH_PPC = "ppc";

	public static final String ARCH_X86_32 = "x86";

	public static final String ARCH_X86_64 = "x86_64";

    // Aarch64 and arm64 are equivalent architectures
    public static final String ARCH_AARCH64 = "aarch64";
    
    public static final String ARCH_ARM64 = "arm64";

	public static final Map< String, String > archMap;

	static
	{
		Map< String, String > architectures = new HashMap<>();
		architectures.put( "x86", ARCH_X86_32 );
		architectures.put( "i386", ARCH_X86_32 );
		architectures.put( "i486", ARCH_X86_32 );
		architectures.put( "i586", ARCH_X86_32 );
		architectures.put( "i686", ARCH_X86_32 );
		architectures.put( "x86_64", ARCH_X86_64 );
		architectures.put( "amd64", ARCH_X86_64 );
		architectures.put( "powerpc", ARCH_PPC );
        architectures.put("arm64", ARCH_ARM64);
        architectures.put("aarch64", ARCH_ARM64);
		archMap = Collections.unmodifiableMap( architectures );
	}
    
    private static String DETECT_CHIP_TERMINAL_COMMAND = "uname -m";

	private String os;

	private String arch;
	
    private boolean rosetta = false;

	/**
	 * Creates a platform detection using the local platform.
	 */
	public PlatformDetection()
	{
		// resolve OS
		if ( System.getProperty("os.name").toLowerCase().contains("win") )
		{
			this.setOs( OS_WINDOWS );
		}
		else if ( System.getProperty("os.name").toLowerCase().replace(" ", "").contains("macosx") )
		{
			this.setOs( OS_OSX );
		}
		else if ( System.getProperty("os.name").toLowerCase().contains("sunos") 
        		|| System.getProperty("os.name").toLowerCase().endsWith("solaris") )
		{
			this.setOs( OS_SOLARIS );
		}
		else if ( System.getProperty("os.name").toLowerCase().contains("linux") 
        		|| System.getProperty("os.name").toLowerCase().endsWith("ix") )
		{
			this.setOs( OS_LINUX );
		}
		else
		{
			throw new IllegalArgumentException( "Unknown operating system " + System.getProperty("os.name") );
		}

		// resolve architecture
		this.setArch( archMap.get( System.getProperty("os.arch") ) );
		if ( this.arch == null )
		{ 
			throw new IllegalArgumentException( "Unknown architecture " + System.getProperty("os.arch") ); 
		}
        if (this.arch.equals(ARCH_X86_64)) {
			try {
				Process proc = Runtime.getRuntime().exec(
						new String[] {"bash", "-c", DETECT_CHIP_TERMINAL_COMMAND});
				String txt = waitProcessExecutionAndGetOutputText(proc);
	    		if (txt.toLowerCase().equals(ARCH_ARM64) || txt.toLowerCase().equals(ARCH_ARM64))
	    			rosetta = true;
			} catch (IOException e) {
				e.printStackTrace();
				System.out.println("Error checking the chip architecture with bash");
			}
        }
	}
	
	public static void main(String[] args) {
		new PlatformDetection();
	}

	/**
	 * @return The operating system of the platform. e.g. windows, linux,
	 *         macosx, etc.
	 */
	public String getOs()
	{
		return os;
	}

	/**
	 * The Operating System of this detection.
	 * 
	 * @param os
	 *            The operating system to be assigned.
	 */
	public void setOs( String os )
	{
		this.os = os;
	}

	/**
	 * @return The system architecture. e.g. x86, x86_64, amd64, etc.
	 */
	public String getArch()
	{
		return arch;
	}

	/**
	 * Sets the system architecture of this detection.
	 * 
	 * @param arch
	 *            The system architecture to be assigned.
	 */
	public void setArch( String arch )
	{
		this.arch = arch;
	}

	@Override
	public String toString()
	{

		return os + "-" + arch;
	}
    
    public boolean isUsingRosseta() {
    	return this.rosetta;
    }
	
	public String waitProcessExecutionAndGetOutputText(Process proc) throws IOException {
		BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(proc.getInputStream()));
		String inputStreamTxt = "";
		boolean first = true;
		while (proc.isAlive() || first) {
			first = false;
			try {
				String txtAux1 = readBufferedReaderIntoStringIntoString(bufferedReader);
				inputStreamTxt += txtAux1;
				Thread.sleep(300);
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
    
    public static boolean isWindows() {
    	return new PlatformDetection().getOs().equals(PlatformDetection.OS_WINDOWS);
    }
    
    public static boolean isWindows(PlatformDetection platform) {
    	return platform.getOs().equals(PlatformDetection.OS_WINDOWS);
    }
    
    public static boolean isLinux() {
    	return new PlatformDetection().getOs().equals(PlatformDetection.OS_LINUX);
    }
    
    public static boolean isLinux(PlatformDetection platform) {
    	return platform.getOs().equals(PlatformDetection.OS_LINUX);
    }
    
    public static boolean isMacOS() {
    	return new PlatformDetection().getOs().equals(PlatformDetection.OS_OSX);
    }
    
    public static boolean isMacOS(PlatformDetection platform) {
    	return platform.getOs().equals(PlatformDetection.OS_OSX);
    }
}
