/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.versionmanagement;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.download.DownloadModel;
import io.bioimage.modelrunner.utils.Constants;


/**
 * An available tensor flow version with its properties and needed artifact URLs.
 * 
 * @author Daniel Felipe Gonzalez Obando and Carlos Javier Garcia Lopez de Haro
 */
public class DeepLearningVersion
{
	private String framework;
    private String version;
    private String pythonVersion;
    private String os;
    private boolean cpu;
    private boolean gpu;
    private boolean rosetta;
    private List<String> jars;
    private String allEnginesDir;
    private String engineName;
    private int minJavaVersion = 8;;
    public static String cpuKey = "cpu";
    public static String gpuKey = "gpu";
    
    /**
     * Create a DeepLearningVersion from the String corresponding to
     * an engine folder name.
     * The folder name has the following format:
     * pytorch-1.7.1-1.7.1-windows-x86_64-cpu
     * where the first word means the engine, the next version number is the python version
     * number, the next version number is the Java JAr version number, then it comes the
     * operating system and then either cpu, gpu or mkl
     * @param engineDir
     * 	the engine folder 
     * @return a {@link #DeepLearningVersion()}
     * @throws IOException if the engine name does not follow the naming convention or the file does not exist
     * @throws IllegalStateException if there are two engine names sharing the same properties. This should be impossible,
     * 	and implies that the user has manipulated wrongly the engines
     */
    public static DeepLearningVersion fromFile(File engineDir) throws IOException, IllegalStateException {
    	DeepLearningVersion dlVersion =  new DeepLearningVersion();
    	dlVersion.engineName = engineDir.getName();
    	dlVersion.allEnginesDir = engineDir.getParentFile().getAbsolutePath();
    	String[] fields = dlVersion.engineName.split("-");
    	dlVersion.setFramework(fields[0]);
    	dlVersion.setPythonVersion(fields[1]);
    	dlVersion.setVersion(fields[2]);
    	dlVersion.setOs(fields[3] + "-" + fields[4]);
    	if (fields.length == 5) {
    		dlVersion.setCPU(false);
    		dlVersion.setGPU(false);
    	} else if (fields.length == 6 && fields[5].toLowerCase().equals(cpuKey)) {
    		dlVersion.setCPU(true);
    		dlVersion.setGPU(false);
    	} else if (fields.length == 6 && fields[5].toLowerCase().equals(gpuKey)) {
    		dlVersion.setCPU(false);
    		dlVersion.setGPU(true);
    	} else if (fields.length == 7 && fields[5].toLowerCase().equals(cpuKey)
    			&& fields[6].toLowerCase().equals(gpuKey)) {
    		dlVersion.setCPU(true);
    		dlVersion.setGPU(true);
    	} else {
    		throw new IOException("The name of the engine does not follow "
    				+ "the engine name convention followed by JDLL: <name_of_the_engine>-"
    				+ "<engine_python_version>-<engine_java_version>-<os>-<cpu_if_supported>-"
    				+ "<gpu_if_supported>: " + engineDir.getName());
    	}
    	List<DeepLearningVersion> candidateVersions = dlVersion.getCandidates();
    	// If the resources file "availableDLVersions.json" is correct, there will only be
        // one coincidence in the list
        if (candidateVersions.size() != 1)
        	throw new IllegalStateException("There should only one engine in the resources engine specs file '"
        			+ Constants.ENGINES_LINK + "' that corresponds to the engine defined by: " 
        			+ dlVersion.engineName);
    	dlVersion.rosetta = candidateVersions.get(0).rosetta;
    	dlVersion.minJavaVersion = candidateVersions.get(0).minJavaVersion;
    	dlVersion.setJars(candidateVersions.get(0).getJars()); 
    	return dlVersion;
    }
    
    /**
     * Get list of versions specified in the resources file containg all the versions
     * that can refer to the version defined by the current instance. Theoretically, if
     * the version is well defined tehre should only be one.
     * @return the number of candidates that can define the wanted version, there should only be one
     */
    private List<DeepLearningVersion> getCandidates() {
    	List<DeepLearningVersion> availableVersions = AvailableEngines.getAll();
        // To find the wanted version compare everything but the JAR files
    	List<DeepLearningVersion> versionsOfInterest = availableVersions.stream()
        	.filter(v -> v.getFramework().toLowerCase().equals(getFramework().toLowerCase()) 
        			&& v.getPythonVersion().toLowerCase().equals(getPythonVersion().toLowerCase())
        			&& v.getVersion().toLowerCase().equals(getVersion().toLowerCase())
        			&& v.getOs().toLowerCase().equals(getOs().toLowerCase())
        			&& v.getCPU() == getCPU() && v.getGPU() == getGPU()
        			)
        	.collect(Collectors.toList());
    	return versionsOfInterest;
    }
    
    /**
     * Returns a list with all the missing Jars for a Deep Learning engine version
     * @return the list with all the missing JAR files
     */
    public List<String> checkMissingJars() {
    	String[] jarsArr = new File(this.allEnginesDir, folderName()).list();
    	List<String> folderJars = Arrays.asList(jarsArr);
    	List<String> missingJars = getJarsFileNames().stream()
    			.filter(jar -> !folderJars.contains(jar) 
    					|| new File(allEnginesDir + File.separator + folderName() + File.separator + jar).length() <= 0)
    			.collect(Collectors.toList());
    	return missingJars;
    }
    
    /**
     * Check whether the provided file path corresponds to a JAR that can be found
     * in the particular DL engine
     * @param jarDir
     * 	path to the JAR of interest
     * @return true if it belongs or false otherwise
     */
    public boolean doesJarBelongToEngine(String jarDir) {
    	File jarFile = new File(jarDir);
    	if (!jarFile.isFile())
    		return false;
    	String jarName = jarFile.getName();
    	return this.jars.stream().anyMatch(i -> i.contains(jarName));
    }
    
    /**
     * Creates the name that the folder containing this Deep Learning version is going to have
     * @return the name of the folder containing the JARs 
     */
    public String folderName() {
    	if (this.engineName == null)
    		engineName = getFramework() + "-" + getPythonVersion() + "-" + getVersion() + "-" + getOs()
                + (getCPU() ? "-cpu" : "") + (getGPU() ? "-gpu" : "");
    	return engineName;
    }

    /**
     * @return whether GPU is supported by the version or not
     */
    public boolean getGPU()
    {
        return gpu;
    }

    /**
     * @param gpu whether GPU is supported by the version or not
     */
    private void setGPU(boolean gpu)
    {
        this.gpu = gpu;
    }

    /**
     * @return The API engine.
     */
    public String getFramework()
    {
        return framework;
    }

    /**
     * @param framework
     *        The API engine
     */
    private void setFramework(String framework)
    {
        this.framework = framework;
    }

    /**
     * @return The API version.
     */
    public String getVersion()
    {
        return version;
    }

    /**
     * @param version
     *        The API version
     */
    private void setVersion(String version)
    {
        this.version = version;
    }

    /**
     * @return The Python library version.
     */
    public String getPythonVersion()
    {
        return pythonVersion;
    }

    /**
     * @param pythonVersion
     *        The Python library version.
     */
    private void setPythonVersion(String pythonVersion)
    {
        this.pythonVersion = pythonVersion;
    }

    /**
     * @return The target operating system.
     */
    public String getOs()
    {
        return os;
    }

    /**
     * @return The target operating system in a short String
     */
    public String getOsShort()
    {
    	if (os.contains("windows"))
    		return "windows";
    	else if (os.contains("linux") || os.contains("unix"))
    		return "linux";
    	else if (os.contains("mac"))
    		return "macos";
    	else
	    	return os;
    }

    /**
     * @param os
     *        The target operating system.
     */
    private void setOs(String os)
    {
        this.os = os;
    }

    /**
     * @return whether CPU is supported by the version or not
     */
    public boolean getCPU()
    {
        return cpu;
    }

    /**
     * @param cpu
     *        whether CPU is supported by the version or not
     */
    private void setCPU(boolean cpu)
    {
        this.cpu = cpu;
    }
    
    /**
     * Whether the engine is supported in Rosetta or not
     * @return whether the engine is supported in Rosetta or not
     */
    public boolean getRosetta() {
    	return this.rosetta;
    }
    
    /**
     * 
     * @return the minimum Java version needed to run the DL engine version
     */
    public int getMinJavaVersion() {
    	return minJavaVersion;
    }

    /**
     * @return The list of associated artifacts for this version.
     */
    public List<String> getJars()
    {
        return jars;
    }
    
    /** 
     * GEt the list of JArs but only containing the string corresponding to the file name
     * @return list of strings representing the names of the JARs
     */
    public List<String> getJarsFileNames() {
    	return jars.stream().map(jar -> {
			try {
				return DownloadModel.getFileNameFromURLString(jar);
			} catch (MalformedURLException e) {
				return jar;
			}
		}).collect(Collectors.toList());
    }

    /**
     * @param jars
     *        The list of associated artifacts for this version.
     */
    private void setJars(List<String> jars)
    {
        this.jars = 
        		jars.stream().filter(jar -> jar != null)
        		.collect(Collectors.toList());
    }
    
    /**
     * REturn true if the Java version of the object instance is 
     * bigger than the Java version of the input DeepLearningVersion
     * @param vv
     * 	version to be compared with
     * @return true if the instance Java version is bigger
     */
    public boolean isJavaVersionBigger(DeepLearningVersion vv) {
    	int result = stringVersionComparator(getVersion(), vv.getVersion());
    	if (result == -1)
    		return true;
    	else
    		return false;
    }
    
    /**
     * REturn true if the Python version of the object instance is 
     * bigger than the Python version of the input DeepLearningVersion
     * @param vv
     * 	version to be compared with
     * @return true if the instance Python version is bigger
     */
    public boolean isPythonVersionBigger(DeepLearningVersion vv) {
    	int result = stringVersionComparator(getPythonVersion(), vv.getPythonVersion());
    	if (result == -1)
    		return true;
    	else
    		return false;
    }
    
    /**
     * This method compares to versions expressed as Strings. Returns 
     * -1 if the first version is bigger, 1 if the second argument is
     * bigger or 0 if the versions are equal.
     * Examples: 2.1 &gt; 2.0; 3.1.2 &lt; 3.2; 3.1.2 &gt; 3.1
     * @param v1
     * 	first version
     * @param v2
     * 	second version
     * @return -1 if the first version is bigger, 1 if the second argument is
     * bigger or 0 if the versions are equal.
     */
    public static int stringVersionComparator(String v1, String v2) {
    	if (v1 == null && v2 == null) 
    		throw new IllegalArgumentException("Both arguments cannot be null.");
    	else if (v1 == null)
    		return 1;
    	else if (v2 == null)
    		return -1;
    	String[] v1Arr = v1.split("\\.");
    	String[] v2Arr = v2.split("\\.");
    	int minNVersions = v1Arr.length > v2Arr.length ? v2Arr.length : v1Arr.length;
    	for (int i = 0; i < minNVersions; i ++) {
    		if (Integer.parseInt(v1Arr[i]) > Integer.parseInt(v2Arr[i])) {
    			return -1;
    		} else if (Integer.parseInt(v1Arr[i]) < Integer.parseInt(v2Arr[i])) {
    			return 1;
    		}
    	}
    	if (v1Arr.length > v2Arr.length)
    		return -1;
    	else if (v1Arr.length < v2Arr.length)
    		return 1;
    	return 0;    		
    }

    /**
     * The string representation is {@code "TensorFlowVersion[version=x, tensorFlowVersion=y, os=z, mode=u, jars=[...]"}.
     */
    @Override
    public String toString()
    {
        return framework + " [version=" + version + ", pythonVersion=" + pythonVersion + ", os=" + os
                + ", cpu=" + cpu + ", gpu=" + gpu + ", jars=" + jars
                + ", rosetta=" + rosetta+ ", minJavaVersion=" + minJavaVersion + "]";
    }

    @Override
    public int hashCode()
    {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((framework == null) ? 0 : framework.hashCode());
        result = prime * result + (cpu ? 0 : "cpu".hashCode());
        result = prime * result + ((os == null) ? 0 : os.hashCode());
        result = prime * result + ((version == null) ? 0 : version.hashCode());
        result = prime * result + (gpu ? 0 : "gpu".hashCode());
        result = prime * result + (rosetta ? 0 : "rosetta".hashCode());
        result = prime * result + minJavaVersion;
        return result;
    }

    @Override
    public boolean equals(Object obj)
    {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        DeepLearningVersion other = (DeepLearningVersion) obj;
        if (framework == null)
        {
            if (other.framework != null)
                return false;
        }
        else if (!framework.equals(other.framework))
            return false;
        if (cpu != other.cpu)
            return false;
        if (os == null)
        {
            if (other.os != null)
                return false;
        }
        else if (!os.equals(other.os))
            return false;
        if (version == null)
        {
            if (other.version != null)
                return false;
        }
        else if (!version.equals(other.version))
            return false;
        if (gpu != other.gpu)
            return false;
        return true;
    }

    /**
     * Return the directory where all the engines are installed
     */
	public void setEnginesDir() {
		allEnginesDir = InstalledEngines.getEnginesDir();
	}

}
