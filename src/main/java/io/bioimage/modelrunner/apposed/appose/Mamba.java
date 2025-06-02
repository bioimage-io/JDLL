/*******************************************************************************
 * Copyright (C) 2021, Ko Sugawara
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/
package io.bioimage.modelrunner.apposed.appose;

import org.apache.commons.compress.archivers.ArchiveException;

import com.sun.jna.Platform;

import io.bioimage.modelrunner.apposed.appose.CondaException.EnvironmentExistsException;
import io.bioimage.modelrunner.download.FileDownloader;
import io.bioimage.modelrunner.system.PlatformDetection;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
/**
 * Python environment manager, implemented by delegating to micromamba.
 * 
 * @author Ko Sugawara
 * @author Curtis Rueden
 */
public class Mamba {
	
	/**
	 * String containing the path that points to the micromamba executable
	 */
	final String mambaCommand;
	/**
	 * Name of the environment where the changes are going to be applied
	 */
	private String envName;
	/**
	 * Root directory of micromamba that also contains the environments folder
	 * 
	 * <pre>
	 * rootdir
	 * ├── bin
	 * │   ├── micromamba(.exe)
	 * │   ... 
	 * ├── envs
	 * │   ├── your_env
	 * │   │   ├── python(.exe)
	 * </pre>
	 */
	private final String rootdir;
	/**
	 * Path to the folder that contains the directories
	 */
	private final String envsdir;
	/**
	 * Whether Micromamba is installed or not
	 */
	private boolean installed = false;
	/**
	 * Progress made on the download from the Internet of the micromamba software. VAlue between 0 and 1.
	 * 
	 */
	private Double mambaDnwldProgress = 0.0;
	/**
	 * Progress made on the decompressing the micromamba files downloaded from the Internet of the micromamba 
	 * software. VAlue between 0 and 1.
	 */
	private Double mambaDecompressProgress = 0.0;
	/**
	 * Consumer that tracks the progress in the download of micromamba, the software used 
	 * by this class to manage Python environments
	 */
	private Consumer<Double> mambaDnwldProgressConsumer = this::updateMambaDnwldProgress;
	/**
	 * Consumer that tracks the progress decompressing the downloaded micromamba files.
	 */
	private Consumer<Double> mambaDecompressProgressConsumer = this::updateMambaDecompressProgress;
	/**
	 * String that contains all the console output produced by micromamba ever since the {@link Mamba} was instantiated
	 */
	private String mambaConsoleOut = "";
	/**
	 * String that contains all the error output produced by micromamba ever since the {@link Mamba} was instantiated
	 */
	private String mambaConsoleErr = "";
	/**
	 * User custom consumer that tracks the console output produced by the micromamba process when it is executed.
	 */
	private Consumer<String> customConsoleConsumer;
	/**
	 * User custom consumer that tracks the error output produced by the micromamba process when it is executed.
	 */
	private Consumer<String> customErrorConsumer;
	/**
	 * Consumer that tracks the console output produced by the micromamba process when it is executed.
	 * This consumer saves all the log of every micromamba execution
	 */
	private Consumer<String> consoleConsumer = this::updateConsoleConsumer;
	/**
	 * Consumer that tracks the error output produced by the micromamba process when it is executed.
	 * This consumer saves all the log of every micromamba execution
	 */
	private Consumer<String> errConsumer = this::updateErrorConsumer;
	/*
	 * Path to Python executable from the environment directory
	 */
	final static String PYTHON_COMMAND = PlatformDetection.isWindows() ? "python.exe" : "bin/python";
	/**
	 * Default name for a Python environment
	 */
	public final static String DEFAULT_ENVIRONMENT_NAME = "base";
	/**
	 * Relative path to the micromamba executable from the micromamba {@link #rootdir}
	 */
	private final static String MICROMAMBA_RELATIVE_PATH = PlatformDetection.isWindows() ? 
			 File.separator + "Library" + File.separator + "bin" + File.separator + "micromamba.exe" 
			: File.separator + "bin" + File.separator + "micromamba";
	/**
	 * Path where Appose installs Micromamba by default
	 */
	final public static String BASE_PATH = Paths.get(System.getProperty("user.home"), ".local", "share", "appose", "micromamba").toString();
	/**
	 * Name of the folder inside the {@link #rootdir} that contains the different Python environments created by the Appose Micromamba
	 */
	final public static String ENVS_NAME = "envs";
	/**
	 * URL from where Micromamba is downloaded to be installed
	 */
	public final static String MICROMAMBA_URL =
		"https://micro.mamba.pm/api/micromamba/" + microMambaPlatform() + "/latest";
	/**
	 * ID used to identify the text retrieved from the error stream when a consumer is used
	 */
	public final static String ERR_STREAM_UUUID = UUID.randomUUID().toString();

	/**
	 * 
	 * @return a String that identifies the current OS to download the correct Micromamba version
	 */
	private static String microMambaPlatform() {
		String osName = System.getProperty("os.name");
		if (osName.startsWith("Windows")) osName = "Windows";
		String osArch = System.getProperty("os.arch");
		switch (osName + "|" + osArch) {
			case "Linux|amd64":      return "linux-64";
			case "Linux|aarch64":    return "linux-aarch64";
			case "Linux|ppc64le":    return "linux-ppc64le";
			case "Mac OS X|x86_64":  return "osx-64";
			case "Mac OS X|aarch64": return "osx-arm64";
			case "Windows|amd64":    return "win-64";
			default:                 return null;
		}
	}
	
	private void updateMambaDnwldProgress(Double pp) {
	    double progress = pp != null ? pp : 0.0;
	    mambaDnwldProgress = progress * 1.0;
	}
	
	private void updateConsoleConsumer(String str) {
		if (str == null) str = "";
		mambaConsoleOut += str;
		if (customConsoleConsumer != null)
			customConsoleConsumer.accept(str);
	}
	
	private void updateErrorConsumer(String str) {
		if (str == null) str = "";
		mambaConsoleErr += str;
		if (customErrorConsumer != null)
			customErrorConsumer.accept(str);
	}
	
	private void updateMambaDecompressProgress(Double pp) {
	    double progress = pp != null ? pp : 0.0;
	    this.mambaDecompressProgress = progress * 1.0;
	}

	/**
	 * Returns a {@link ProcessBuilder} with the working directory specified in the
	 * constructor.
	 * 
	 * @param isInheritIO
	 *            Sets the source and destination for subprocess standard I/O to be
	 *            the same as those of the current Java process.
	 * @return The {@link ProcessBuilder} with the working directory specified in
	 *         the constructor.
	 */
	private ProcessBuilder getBuilder( final boolean isInheritIO )
	{
		final ProcessBuilder builder = new ProcessBuilder().directory( new File( rootdir ) );
		if ( isInheritIO )
			builder.inheritIO();
		return builder;
	}

	/**
	 * Create a new {@link Mamba} object. The root dir for the Micromamba installation
	 * will be the default base path defined at {@link BASE_PATH}
	 * If there is no Micromamba found at the base path {@link BASE_PATH}, a {@link MambaInstallException} will be thrown
	 * 
	 * It is expected that the Micromamba installation has executable commands as shown below:
	 * 
	 * <pre>
	 * MAMBA_ROOT
	 * ├── bin
	 * │   ├── micromamba(.exe)
	 * │   ... 
	 * ├── envs
	 * │   ├── your_env
	 * │   │   ├── python(.exe)
	 * </pre>
	 * 
	 */
	public Mamba() {
		this(BASE_PATH);
	}

	/**
	 * Create a new Conda object. The root dir for Conda installation can be
	 * specified as {@code String}. 
	 * If there is no Micromamba found at the specified path, it will be installed automatically 
	 * if the parameter 'installIfNeeded' is true. If not a {@link MambaInstallException} will be thrown.
	 * 
	 * It is expected that the Conda installation has executable commands as shown below:
	 * 
	 * <pre>
	 * MAMBA_ROOT
	 * ├── bin
	 * │   ├── micromamba(.exe)
	 * │   ... 
	 * ├── envs
	 * │   ├── your_env
	 * │   │   ├── python(.exe)
	 * </pre>
	 * 
	 * @param rootdir
	 *  The root dir for Mamba installation.
	 */
	public Mamba( final String rootdir) {
		if (rootdir == null)
			this.rootdir = BASE_PATH;
		else
			this.rootdir = rootdir;
		this.mambaCommand = new File(this.rootdir + MICROMAMBA_RELATIVE_PATH).getAbsolutePath();
		this.envsdir = Paths.get(rootdir, ENVS_NAME).toAbsolutePath().toString();
		boolean filesExist = Files.notExists( Paths.get( mambaCommand ) );
		if (!filesExist)
			return;
		try {
			getVersion();
		} catch (Exception ex) {
			return;
		}
		installed = true;
	}
	
	/**
	 * Check whether micromamba is installed or not to be able to use the instance of {@link Mamba}
	 * @return whether micromamba is installed or not to be able to use the instance of {@link Mamba}
	 */
	public boolean checkMambaInstalled() {
		try {
			getVersion();
			this.installed = true;
		} catch (Exception ex) {
			this.installed = false;
			return false;
		}
		return true;
	}
	
	/**
	 * 
	 * @return the progress made on the download from the Internet of the micromamba software. VAlue between 0 and 1.
	 */
	public double getMicromambaDownloadProgress(){
		return this.mambaDnwldProgress;
	}
	
	/**
	 * 
	 * @return the the progress made on the decompressing the micromamba files downloaded from the Internet of the micromamba 
	 * 	software. VAlue between 0 and 1.
	 */
	public double getMicromambaDecompressProgress(){
		return this.mambaDecompressProgress;
	}
	
	/**
	 * 
	 * @return all the console output produced by micromamba ever since the {@link Mamba} was instantiated
	 */
	public String getMicromambaConsoleStream(){
		return this.mambaConsoleOut;
	}
	
	/**
	 * 
	 * @return all the error output produced by micromamba ever since the {@link Mamba} was instantiated
	 */
	public String getMicromambaErrStream(){
		return mambaConsoleErr;
	}
	
	/**
	 * Set a custom consumer for the console output of every micromamba call
	 * @param custom
	 * 	custom consumer that receives every console line outputed by ecery micromamba call
	 */
	public void setConsoleOutputConsumer(Consumer<String> custom) {
		this.customConsoleConsumer = custom;
	}
	
	/**
	 * Set a custom consumer for the error output of every micromamba call
	 * @param custom
	 * 	custom consumer that receives every error line outputed by ecery micromamba call
	 */
	public void setErrorOutputConsumer(Consumer<String> custom) {
		this.customErrorConsumer = custom;
	}
	
	private File tempDirMacos() throws IOException, URISyntaxException {
		
        String filename = "micromamba-" + UUID.randomUUID() + ".tar.bz2";
        File tempFile = new File(BASE_PATH, filename);
        boolean created = tempFile.createNewFile();
        if (!created) {
            throw new IOException("Failed to create temp file: " + tempFile.getAbsolutePath());
        }
        tempFile.deleteOnExit();
		return tempFile;
	}
	
	private File downloadMicromamba() throws IOException, URISyntaxException {
		final File tempFile;
		if (PlatformDetection.isMacOS())
			tempFile = tempDirMacos();
		else
			tempFile = File.createTempFile( "micromamba", ".tar.bz2" );
		tempFile.deleteOnExit();
		URL website = FileDownloader.redirectedURL(new URL(MICROMAMBA_URL));
		Consumer<Double> micromambaConsumer = (d) -> {
			d = (double) (Math.round(d * 1000) / 10);
			customConsoleConsumer.accept("Installing micromamba: " + d + "%");
		};
		FileDownloader fd = new FileDownloader(website.toString(), tempFile);
		fd.setPartialProgressConsumer(micromambaConsumer);
		long size = fd.getOnlineFileSize();
		try {
			fd.download(Thread.currentThread());
		} catch (ExecutionException e) {
			throw new RuntimeException(e);
		};
		if ((((double) tempFile.length()) / ((double) size)) < 1)
			throw new IOException("Error downloading micromamba from: " + MICROMAMBA_URL);
		return tempFile;
	}
	
	private void decompressMicromamba(final File tempFile) 
				throws FileNotFoundException, IOException, ArchiveException, InterruptedException {
		final File tempTarFile = File.createTempFile( "micromamba", ".tar" );
		tempTarFile.deleteOnExit();
		MambaInstallerUtils.unBZip2(tempFile, tempTarFile);
		File mambaBaseDir = new File(rootdir);
		if (!mambaBaseDir.isDirectory() && !mambaBaseDir.mkdirs())
	        throw new IOException("Failed to create Micromamba default directory " + mambaBaseDir.getParentFile().getAbsolutePath()
	        		+ ". Please try installing it in another directory.");
		MambaInstallerUtils.unTar(tempTarFile, mambaBaseDir);
		if (!(new File(envsdir)).isDirectory() && !new File(envsdir).mkdirs())
	        throw new IOException("Failed to create Micromamba default envs directory " + envsdir);
		boolean executableSet = new File(mambaCommand).setExecutable(true);
		if (!executableSet)
			throw new IOException("Cannot set file as executable due to missing permissions, "
					+ "please do it manually: " + mambaCommand);
	}
	
	/**
	 * Install Micromamba automatically
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws ArchiveException if there is any error decompressing
	 * @throws URISyntaxException  if there is any error with the micromamba url
	 */
	public void installMicromamba() throws IOException, InterruptedException, ArchiveException, URISyntaxException {
		checkMambaInstalled();
		if (installed) return;
		decompressMicromamba(downloadMicromamba());
		checkMambaInstalled();
	}
	
	public String getEnvsDir() {
		return this.envsdir;
	}

	/**
	 * Returns {@code \{"cmd.exe", "/c"\}} for Windows and an empty list for
	 * Mac/Linux.
	 * 
	 * @return {@code \{"cmd.exe", "/c"\}} for Windows and an empty list for
	 *         Mac/Linux.
	 * @throws IOException
	 */
	private static List< String > getBaseCommand()
	{
		final List< String > cmd = new ArrayList<>();
		if ( PlatformDetection.isWindows() )
			cmd.addAll( Arrays.asList( "cmd.exe", "/c" ) );
		return cmd;
	}

	/**
	 * Run {@code conda update} in the activated environment. A list of packages to
	 * be updated and extra parameters can be specified as {@code args}.
	 * 
	 * @param args
	 *            The list of packages to be updated and extra parameters as
	 *            {@code String...}.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void update( final String... args ) throws IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		updateIn( envName, args );
	}

	/**
	 * Run {@code conda update} in the specified environment. A list of packages to
	 * update and extra parameters can be specified as {@code args}.
	 * 
	 * @param envName
	 *            The environment name to be used for the update command.
	 * @param args
	 *            The list of packages to be updated and extra parameters as
	 *            {@code String...}.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void updateIn( final String envName, final String... args ) throws IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		final List< String > cmd = new ArrayList<>( Arrays.asList( "update", "-p", this.envsdir + File.separator + envName ) );
		cmd.addAll( Arrays.asList( args ) );
		if (!cmd.contains("--yes") && !cmd.contains("-y")) cmd.add("--yes");
		runMamba( cmd.stream().toArray( String[]::new ) );
	}

	/**
	 * Run {@code conda create} to create a conda environment defined by the input environment yaml file.
	 * 
	 * @param envName
	 *            The environment name to be created.
	 * @param envYaml
	 *            The environment yaml file containing the information required to build it 
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void createWithYaml( final String envName, final String envYaml ) throws IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		createWithYaml(envName, envYaml, false);
	}

	/**
	 * Run {@code conda create} to create a conda environment defined by the input environment yaml file.
	 * 
	 * @param envName
	 *            The environment name to be created. It should not be a path, just the name.
	 * @param envYaml
	 *            The environment yaml file containing the information required to build it  
	 * @param envName
	 *            The environment name to be created.
	 * @param isForceCreation
	 *            Force creation of the environment if {@code true}. If this value
	 *            is {@code false} and an environment with the specified name
	 *            already exists, throw an {@link EnvironmentExistsException}.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws RuntimeException if the process to create the env of the yaml file is not terminated correctly. If there is any error running the commands
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void createWithYaml( final String envName, final String envYaml, final boolean isForceCreation) throws IOException, InterruptedException, RuntimeException, MambaInstallException
	{
		if (envName.contains(File.pathSeparator))
			throw new IllegalArgumentException("The environment name should not contain the file separator character: '"
					+ File.separator + "'");
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		if ( !isForceCreation && getEnvironmentNames().contains( envName ) )
			throw new EnvironmentExistsException();
		runMamba("env", "create", "--prefix",
				envsdir + File.separator + envName, "-f", envYaml, "-y", "-vv" );
		if (this.checkDependencyInEnv(envsdir + File.separator + envName, "python"))
			installApposeFromSource(envsdir + File.separator + envName);
	}

	/**
	 * Run {@code conda create} to create an empty conda environment.
	 * 
	 * @param envName
	 *            The environment name to be created.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void create( final String envName ) throws IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		create( envName, false );
	}

	/**
	 * Run {@code conda create} to create an empty conda environment.
	 * 
	 * @param envName
	 *            The environment name to be created.
	 * @param isForceCreation
	 *            Force creation of the environment if {@code true}. If this value
	 *            is {@code false} and an environment with the specified name
	 *            already exists, throw an {@link EnvironmentExistsException}.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws RuntimeException
	 *             If there is any error running the commands
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void create( final String envName, final boolean isForceCreation ) throws IOException, InterruptedException, RuntimeException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		if ( !isForceCreation && getEnvironmentNames().contains( envName ) )
			throw new EnvironmentExistsException();
		runMamba( "create", "-y", "-p", envsdir + File.separator + envName );
		if (this.checkDependencyInEnv(envsdir + File.separator + envName, "python"))
			installApposeFromSource(envsdir + File.separator + envName);
	}

	/**
	 * Run {@code conda create} to create a new mamba environment with a list of
	 * specified packages.
	 * 
	 * @param envName
	 *            The environment name to be created.
	 * @param args
	 *            The list of packages to be installed on environment creation and
	 *            extra parameters as {@code String...}.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void create( final String envName, final String... args ) throws IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		create( envName, false, args );
	}

	/**
	 * Run {@code conda create} to create a new conda environment with a list of
	 * specified packages.
	 * 
	 * @param envName
	 *            The environment name to be created.
	 * @param isForceCreation
	 *            Force creation of the environment if {@code true}. If this value
	 *            is {@code false} and an environment with the specified name
	 *            already exists, throw an {@link EnvironmentExistsException}.
	 * @param args
	 *            The list of packages to be installed on environment creation and
	 *            extra parameters as {@code String...}.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void create( final String envName, final boolean isForceCreation, final String... args ) throws IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		if ( !isForceCreation && getEnvironmentNames().contains( envName ) )
			throw new EnvironmentExistsException();
		final List< String > cmd = new ArrayList<>( Arrays.asList( "create", "-p", envsdir + File.separator + envName ) );
		cmd.addAll( Arrays.asList( args ) );
		if (!cmd.contains("--yes") && !cmd.contains("-y")) cmd.add("--yes");
		runMamba( cmd.stream().toArray( String[]::new ) );
		if (this.checkDependencyInEnv(envsdir + File.separator + envName, "python"))
			installApposeFromSource(envsdir + File.separator + envName);
	}

	/**
	 * Run {@code conda create} to create a new conda environment with a list of
	 * specified packages.
	 * 
	 * @param envName
	 *            The environment name to be created. CAnnot be null.
	 * @param isForceCreation
	 *            Force creation of the environment if {@code true}. If this value
	 *            is {@code false} and an environment with the specified name
	 *            already exists, throw an {@link EnvironmentExistsException}.
	 * @param channels
	 *            the channels from where the packages can be installed. Can be null
	 * @param packages
	 * 			  the packages that want to be installed during env creation. They can contain the version.
	 * 			  For example, "python" or "python=3.10.1", "numpy" or "numpy=1.20.1". CAn be null if no packages want to be installed
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws RuntimeException
	 *             If there is any error running the commands
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void create( final String envName, final boolean isForceCreation, List<String> channels, List<String> packages ) throws IOException, InterruptedException, RuntimeException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		Objects.requireNonNull(envName, "The name of the environment of interest needs to be provided.");
		if ( !isForceCreation && getEnvironmentNames().contains( envName ) )
			throw new EnvironmentExistsException();
		final List< String > cmd = new ArrayList<>( Arrays.asList( "create", "-p", envsdir + File.separator + envName ) );
		if (channels == null) channels = new ArrayList<String>();
		for (String chan : channels) { cmd.add("-c"); cmd.add(chan);}
		if (packages == null) packages = new ArrayList<String>();
		for (String pack : packages) { cmd.add(pack);}
		if (!cmd.contains("--yes") && !cmd.contains("-y")) cmd.add("--yes");
		runMamba( cmd.stream().toArray( String[]::new ) );
		if (this.checkDependencyInEnv(envsdir + File.separator + envName, "python"))
			installApposeFromSource(envsdir + File.separator + envName);
	}

	/**
	 * This method works as if the user runs {@code conda activate envName}. This
	 * method internally calls {@link Mamba#setEnvName(String)}.
	 * 
	 * @param envName
	 *            The environment name to be activated.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void activate( final String envName ) throws IOException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		if ( getEnvironmentNames().contains( envName ) )
			setEnvName( envName );
		else
			throw new IllegalArgumentException( "environment: " + envName + " not found." );
	}

	/**
	 * This method works as if the user runs {@code conda deactivate}. This method
	 * internally sets the {@code envName} to {@code base}.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void deactivate() throws MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		setEnvName( DEFAULT_ENVIRONMENT_NAME );
	}

	/**
	 * This method is used by {@code Conda#activate(String)} and
	 * {@code Conda#deactivate()}. This method is kept private since it is not
	 * expected to call this method directory.
	 * 
	 * @param envName
	 *            The environment name to be set.
	 */
	private void setEnvName( final String envName )
	{
		this.envName = envName;
	}

	/**
	 * Returns the active environment name.
	 * 
	 * @return The active environment name.
	 * 
	 */
	public String getEnvName()
	{
		return envName;
	}

	/**
	 * Run {@code conda install} in the activated environment. A list of packages to
	 * install and extra parameters can be specified as {@code args}.
	 * 
	 * @param args
	 *            The list of packages to be installed and extra parameters as
	 *            {@code String...}.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void install( final String... args ) throws IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		installIn( envName, args );
	}

	/**
	 * Run {@code conda install} in the activated environment. A list of packages to
	 * install and extra parameters can be specified as {@code args}.
	 * 
	 * @param channels
	 *            the channels from where the packages can be installed. Can be null
	 * @param packages
	 * 			  the packages that want to be installed during env creation. They can contain the version.
	 * 			  For example, "python" or "python=3.10.1", "numpy" or "numpy=1.20.1". CAn be null if no packages want to be installed
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void install( List<String> channels, List<String> packages ) throws IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		installIn( envName, channels, packages );
	}

	/**
	 * Run {@code conda install} in the specified environment. A list of packages to
	 * install and extra parameters can be specified as {@code args}.
	 * 
	 * @param envName
	 *            The environment name to be used for the install command.
	 * @param channels
	 *            the channels from where the packages can be installed. Can be null
	 * @param packages
	 * 			  the packages that want to be installed during env creation. They can contain the version.
	 * 			  For example, "python" or "python=3.10.1", "numpy" or "numpy=1.20.1". CAn be null if no packages want to be installed
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws RuntimeException if the process to create the env of the yaml file is not terminated correctly. If there is any error running the commands
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void installIn( final String envName, List<String> channels, List<String> packages ) throws IOException, InterruptedException, RuntimeException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		Objects.requireNonNull(envName, "The name of the environment of interest needs to be provided.");		
		final List< String > cmd = new ArrayList<>( Arrays.asList( "install", "-y", "-p", this.envsdir + File.separator + envName ) );
		if (channels == null) channels = new ArrayList<String>();
		for (String chan : channels) { cmd.add("-c"); cmd.add(chan);}
		if (packages == null) packages = new ArrayList<String>();
		for (String pack : packages) { cmd.add(pack);}
		runMamba( cmd.stream().toArray( String[]::new ) );
	}

	/**
	 * Run {@code conda install} in the specified environment. A list of packages to
	 * install and extra parameters can be specified as {@code args}.
	 * 
	 * @param envName
	 *            The environment name to be used for the install command.
	 * @param args
	 *            The list of packages to be installed and extra parameters as
	 *            {@code String...}.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void installIn( final String envName, final String... args ) throws IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		final List< String > cmd = new ArrayList<>( Arrays.asList( "install", "-p", this.envsdir + File.separator + envName ) );
		cmd.addAll( Arrays.asList( args ) );
		if (!cmd.contains("--yes") && !cmd.contains("-y")) cmd.add("--yes");
		runMamba( cmd.stream().toArray( String[]::new ) );
	}

	/**
	 * Run {@code pip install} in the activated environment. A list of packages to
	 * install and extra parameters can be specified as {@code args}.
	 * 
	 * @param args
	 *            The list of packages to be installed and extra parameters as
	 *            {@code String...}.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void pipInstall( final String... args ) throws IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		pipInstallIn( envName, args );
	}

	/**
	 * Run {@code pip install} in the specified environment. A list of packages to
	 * install and extra parameters can be specified as {@code args}.
	 * 
	 * @param envName
	 *            The environment name to be used for the install command.
	 * @param args
	 *            The list of packages to be installed and extra parameters as
	 *            {@code String...}.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void pipInstallIn( final String envName, final String... args ) throws IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		final List< String > cmd = new ArrayList<>( Arrays.asList( "-m", "pip", "install" ) );
		cmd.addAll( Arrays.asList( args ) );
		runPythonIn( envName, cmd.stream().toArray( String[]::new ) );
	}

	/**
	 * Run a Python command in the activated environment. This method automatically
	 * sets environment variables associated with the activated environment. In
	 * Windows, this method also sets the {@code PATH} environment variable so that
	 * the specified environment runs as expected.
	 * 
	 * @param args
	 *            One or more arguments for the Python command.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void runPython( final String... args ) throws IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		runPythonIn( envName, args );
	}

	/**
	 * TODO stop process if the thread is interrupted, same as with mamba, look for runmamna method for example
	 * TODO stop process if the thread is interrupted, same as with mamba, look for runmamna method for example
	 * TODO stop process if the thread is interrupted, same as with mamba, look for runmamna method for example
	 * TODO stop process if the thread is interrupted, same as with mamba, look for runmamna method for example
	 * TODO stop process if the thread is interrupted, same as with mamba, look for runmamna method for example
	 * 
	 * Run a Python command in the specified environment. This method automatically
	 * sets environment variables associated with the specified environment. In
	 * Windows, this method also sets the {@code PATH} environment variable so that
	 * the specified environment runs as expected.
	 * 
	 * @param envName
	 *            The environment name used to run the Python command.
	 * @param args
	 *            One or more arguments for the Python command.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void runPythonIn( final String envName, final String... args ) throws IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		final List< String > cmd = getBaseCommand();
		List<String> argsList = new ArrayList<String>();
		String envDir;
		if (new File(envName, PYTHON_COMMAND).isFile()) {
			argsList.add( coverArgWithDoubleQuotes(Paths.get( envName, PYTHON_COMMAND ).toAbsolutePath().toString()) );
			envDir = Paths.get( envName ).toAbsolutePath().toString();
		} else if (Paths.get( this.envsdir, envName, PYTHON_COMMAND ).toFile().isFile()) {
			argsList.add( coverArgWithDoubleQuotes(Paths.get( this.envsdir, envName, PYTHON_COMMAND ).toAbsolutePath().toString()) );
			envDir = Paths.get( envsdir, envName ).toAbsolutePath().toString();
		} else 
			throw new IOException("The environment provided ("
					+ envName + ") does not exist or does not contain a Python executable (" + PYTHON_COMMAND + ").");
		argsList.addAll( Arrays.asList( args ).stream().map(aa -> {
							if (aa.contains(" ") && PlatformDetection.isWindows()) return coverArgWithDoubleQuotes(aa);
							else return aa;
						}).collect(Collectors.toList()) );
		boolean containsSpaces = argsList.stream().filter(aa -> aa.contains(" ")).collect(Collectors.toList()).size() > 0;
		
		if (!containsSpaces || !PlatformDetection.isWindows()) cmd.addAll(argsList);
		else cmd.add(surroundWithQuotes(argsList));

		final ProcessBuilder builder = getBuilder( false );
		if ( PlatformDetection.isWindows() )
		{
			final Map< String, String > envs = builder.environment();
			envs.keySet().removeIf(key ->
			     key.equalsIgnoreCase("PATH")
			  || key.equalsIgnoreCase("PYTHONPATH")
			  || key.equalsIgnoreCase("PYTHONHOME")
			);
			envs.put( "Path", envDir + ";");
			envs.put( "Path", Paths.get( envDir, "Scripts" ).toString() + ";" + envs.get( "Path" ) );
			envs.put( "Path", Paths.get( envDir, "Library" ).toString() + ";" + envs.get( "Path" ) );
			envs.put( "Path", Paths.get( envDir, "Library", "Bin" ).toString() + ";" + envs.get( "Path" ) );
		}
		// TODO find way to get env vars in micromamba builder.environment().putAll( getEnvironmentVariables( envName ) );
		runPythonIn(builder.command( cmd ), this.consoleConsumer, this.errConsumer);
	}

	/**
	 * Run a Python command in the specified environment. This method automatically
	 * sets environment variables associated with the specified environment. In
	 * Windows, this method also sets the {@code PATH} environment variable so that
	 * the specified environment runs as expected.
	 * 
	 * @param envFile
	 *            file corresponding to the environment directory
	 * @param args
	 *            One or more arguments for the Python command.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 */
	public static void runPythonIn( final File envFile, final String... args ) throws IOException, InterruptedException
	{
		if (!Paths.get( envFile.getAbsolutePath(), PYTHON_COMMAND ).toFile().isFile())
			throw new IOException("No Python found in the environment provided. The following "
					+ "file does not exist: " + Paths.get( envFile.getAbsolutePath(), PYTHON_COMMAND ).toAbsolutePath());
		final List< String > cmd = getBaseCommand();
		List<String> argsList = new ArrayList<String>();
		argsList.add( coverArgWithDoubleQuotes(Paths.get( envFile.getAbsolutePath(), PYTHON_COMMAND ).toAbsolutePath().toString()) );
		argsList.addAll( Arrays.asList( args ).stream().map(aa -> {
							if (Platform.isWindows() && aa.contains(" ")) return coverArgWithDoubleQuotes(aa);
							else return aa;
						}).collect(Collectors.toList()) );
		boolean containsSpaces = argsList.stream().filter(aa -> aa.contains(" ")).collect(Collectors.toList()).size() > 0;
		
		if (!containsSpaces || !PlatformDetection.isWindows()) cmd.addAll(argsList);
		else cmd.add(surroundWithQuotes(argsList));
		
		
		final ProcessBuilder builder = new ProcessBuilder().directory( envFile );
		if ( PlatformDetection.isWindows() )
		{
			final Map< String, String > envs = builder.environment();
			envs.keySet().removeIf(key ->
			     key.equalsIgnoreCase("PATH")
			  || key.equalsIgnoreCase("PYTHONPATH")
			  || key.equalsIgnoreCase("PYTHONHOME")
			);
			final String envDir = envFile.getAbsolutePath();
			envs.put( "Path", envDir + ";");
			envs.put( "Path", Paths.get( envDir, "Scripts" ).toString() + ";" + envs.get( "Path" ) );
			envs.put( "Path", Paths.get( envDir, "Library" ).toString() + ";" + envs.get( "Path" ) );
			envs.put( "Path", Paths.get( envDir, "Library", "Bin" ).toString() + ";" + envs.get( "Path" ) );
		}

        Process p = builder.command( cmd ).start();
        Thread reader = new Thread(() -> {
        	try (
        			BufferedReader or = new BufferedReader(new InputStreamReader(p.getInputStream()));
        			BufferedReader er = new BufferedReader(new InputStreamReader(p.getErrorStream()));
        			) {
        		String line = null;
        		String errLine = null;
        		while ((line = or.readLine()) != null || (errLine = er.readLine()) != null ) {
        			if (line != null) System.out.println(line);
        			if (errLine != null) System.out.println(errLine);
        		}
        	}  catch (IOException e) {
        		e.printStackTrace();
        	}
        }, "stdout-stderr-reader");
        reader.setDaemon(true);
        reader.start();

        int exitCode = p.waitFor();
        reader.join();  // wait for final log lines
        if (exitCode != 0)
        	throw new RuntimeException("Error executing the following command: " + builder.command());
        /**
        if ( builder.command( cmd ).start().waitFor() != 0 )
			throw new RuntimeException("Error executing the following command: " + builder.command());
			*/
	}
	
	private static void runPythonIn(ProcessBuilder builder, Consumer<String> consumerOut, Consumer<String> consumerErr) throws RuntimeException, IOException, InterruptedException, MambaInstallException
	{
		Thread mainThread = Thread.currentThread();
		SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss");
		Process process = builder.start();
		// Use separate threads to read each stream to avoid a deadlock.
		consumerOut.accept(sdf.format(Calendar.getInstance().getTime()) + " -- STARTING PROCESS: " + builder.command() + System.lineSeparator());
		long updatePeriod = 300;
		String[] mambaConsoleOut = new String[] {""};
		String[] mambaConsoleErr = new String[] {""};
		Thread outputThread = new Thread(() -> {
			try (
			        InputStream inputStream = process.getInputStream();
			        InputStream errStream = process.getErrorStream();
					){
		        byte[] buffer = new byte[1024]; // Buffer size can be adjusted
		        StringBuilder processBuff = new StringBuilder();
		        StringBuilder errBuff = new StringBuilder();
		        String processChunk = "";
		        String errChunk = "";
                int newLineIndex;
		        long t0 = System.currentTimeMillis();
		        while (process.isAlive() || inputStreamOpen(inputStream)) {
		        	if (mainThread.isInterrupted() || !mainThread.isAlive()) {
		        		process.destroyForcibly();
		        		return;
		        	}
		            if (inputStreamOpen(inputStream)) {
		                processBuff.append(new String(buffer, 0, inputStream.read(buffer)));
		                while ((newLineIndex = processBuff.indexOf(System.lineSeparator())) != -1) {
		                	processChunk += sdf.format(Calendar.getInstance().getTime()) + " -- " 
		                					+ processBuff.substring(0, newLineIndex + 1).trim() + System.lineSeparator();
		                	processBuff.delete(0, newLineIndex + 1);
		                }
		            }
		            if (inputStreamOpen(errStream)) {
		                errBuff.append(new String(buffer, 0, errStream.read(buffer)));
		                while ((newLineIndex = errBuff.indexOf(System.lineSeparator())) != -1) {
		                	errChunk += ERR_STREAM_UUUID + errBuff.substring(0, newLineIndex + 1).trim() + System.lineSeparator();
		                	errBuff.delete(0, newLineIndex + 1);
		                }
		            }
	                // Sleep for a bit to avoid busy waiting
	                Thread.sleep(60);
	                if (System.currentTimeMillis() - t0 > updatePeriod) {
	                	consumerOut.accept(processChunk);
	                	consumerErr.accept(errChunk);
	                	mambaConsoleOut[0] += processChunk + System.lineSeparator();
	                	mambaConsoleErr[0] += errChunk + System.lineSeparator();
						processChunk = "";
						errChunk = "";
						t0 = System.currentTimeMillis();
					}
		        }
		        if (inputStreamOpen(inputStream)) {
	                processBuff.append(new String(buffer, 0, inputStream.read(buffer)));
                	processChunk += sdf.format(Calendar.getInstance().getTime()) + " -- " + processBuff.toString().trim();
	            }
	            if (inputStreamOpen(errStream)) {
	                errBuff.append(new String(buffer, 0, errStream.read(buffer)));
	                errChunk += ERR_STREAM_UUUID + errBuff.toString().trim();
	            }
	            consumerErr.accept(errChunk);
	            consumerOut.accept(processChunk + System.lineSeparator() 
								+ sdf.format(Calendar.getInstance().getTime()) + " -- TERMINATED PROCESS");
		    } catch (IOException | InterruptedException e) {
		        e.printStackTrace();
		    }
		});
		// Start reading threads
		outputThread.start();
		int processResult;
		try {
			processResult = process.waitFor();
		} catch (InterruptedException ex) {
			process.destroyForcibly();
			throw new InterruptedException("Mamba process stopped. The command being executed was: " + builder.command());
		}
		// Wait for all output to be read
		outputThread.join();
		if (processResult != 0)
        	throw new RuntimeException("Error executing the following command: " + builder.command()
        								+ System.lineSeparator() + mambaConsoleOut[0]
        								+ System.lineSeparator() + mambaConsoleErr[0]);
	}

	/**
	 * Returns Conda version as a {@code String}.
	 * 
	 * @return The Conda version as a {@code String}.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public String getVersion() throws IOException, InterruptedException, MambaInstallException
	{
		final List< String > cmd = getBaseCommand();
		if (mambaCommand.contains(" ") && PlatformDetection.isWindows())
			cmd.add( surroundWithQuotes(Arrays.asList( coverArgWithDoubleQuotes(mambaCommand), "--version" )) );
		else
			cmd.addAll( Arrays.asList( coverArgWithDoubleQuotes(mambaCommand), "--version" ) );
		final Process process = getBuilder( false ).command( cmd ).start();
		if ( process.waitFor() != 0 )
			throw new RuntimeException("Error getting Micromamba version");
		return new BufferedReader( new InputStreamReader( process.getInputStream() ) ).readLine();
	}

	/**
	 * Run a Conda command with one or more arguments.
	 * 
	 * @param isInheritIO
	 *            Sets the source and destination for subprocess standard I/O to be
	 *            the same as those of the current Java process.
	 * @param args
	 *            One or more arguments for the Mamba command.
	 * @throws RuntimeException
	 *             If there is any error running the commands
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void runMamba(boolean isInheritIO, final String... args ) throws RuntimeException, IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		Thread mainThread = Thread.currentThread();
		SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss");
		
		final List< String > cmd = getBaseCommand();
		List<String> argsList = new ArrayList<String>();
		argsList.add( coverArgWithDoubleQuotes(mambaCommand) );
		argsList.addAll( Arrays.asList( args ).stream().map(aa -> {
			if (aa.contains(" ") && PlatformDetection.isWindows()) return coverArgWithDoubleQuotes(aa);
			else return aa;
		}).collect(Collectors.toList()) );
		boolean containsSpaces = argsList.stream().filter(aa -> aa.contains(" ")).collect(Collectors.toList()).size() > 0;
		
		if (!containsSpaces || !PlatformDetection.isWindows()) cmd.addAll(argsList);
		else cmd.add(surroundWithQuotes(argsList));
		
		ProcessBuilder builder = getBuilder(isInheritIO).command(cmd);
		Process process = builder.start();
		// Use separate threads to read each stream to avoid a deadlock.
		this.consoleConsumer.accept(sdf.format(Calendar.getInstance().getTime()) + " -- STARTING INSTALLATION" + System.lineSeparator());
		long updatePeriod = 300;
		Thread outputThread = new Thread(() -> {
			try (
			        InputStream inputStream = process.getInputStream();
			        InputStream errStream = process.getErrorStream();
					){
		        byte[] buffer = new byte[1024]; // Buffer size can be adjusted
		        StringBuilder processBuff = new StringBuilder();
		        StringBuilder errBuff = new StringBuilder();
		        String processChunk = "";
		        String errChunk = "";
                int newLineIndex;
		        long t0 = System.currentTimeMillis();
		        while (process.isAlive() || inputStreamOpen(inputStream)) {
		        	if (mainThread.isInterrupted() || !mainThread.isAlive()) {
		        		process.destroyForcibly();
		        		return;
		        	}
		            if (inputStreamOpen(inputStream)) {
		                processBuff.append(new String(buffer, 0, inputStream.read(buffer)));
		                while ((newLineIndex = processBuff.indexOf(System.lineSeparator())) != -1) {
		                	processChunk += sdf.format(Calendar.getInstance().getTime()) + " -- " 
		                					+ processBuff.substring(0, newLineIndex + 1).trim() + System.lineSeparator();
		                	processBuff.delete(0, newLineIndex + 1);
		                }
		            }
		            if (inputStreamOpen(errStream)) {
		                errBuff.append(new String(buffer, 0, errStream.read(buffer)));
		                while ((newLineIndex = errBuff.indexOf(System.lineSeparator())) != -1) {
		                	errChunk += ERR_STREAM_UUUID + errBuff.substring(0, newLineIndex + 1).trim() + System.lineSeparator();
		                	errBuff.delete(0, newLineIndex + 1);
		                }
		            }
	                // Sleep for a bit to avoid busy waiting
	                Thread.sleep(60);
	                if (System.currentTimeMillis() - t0 > updatePeriod) {
						this.consoleConsumer.accept(processChunk);
						this.consoleConsumer.accept(errChunk);
						processChunk = "";
						errChunk = "";
						t0 = System.currentTimeMillis();
					}
		        }
		        if (inputStreamOpen(inputStream)) {
	                processBuff.append(new String(buffer, 0, inputStream.read(buffer)));
                	processChunk += sdf.format(Calendar.getInstance().getTime()) + " -- " + processBuff.toString().trim();
	            }
	            if (inputStreamOpen(errStream)) {
	                errBuff.append(new String(buffer, 0, errStream.read(buffer)));
	                errChunk += ERR_STREAM_UUUID + errBuff.toString().trim();
	            }
				this.errConsumer.accept(errChunk);
				this.consoleConsumer.accept(processChunk + System.lineSeparator() 
								+ sdf.format(Calendar.getInstance().getTime()) + " -- TERMINATED PROCESS");
		    } catch (IOException | InterruptedException e) {
		        e.printStackTrace();
		    }
		});
		// Start reading threads
		outputThread.start();
		int processResult;
		try {
			processResult = process.waitFor();
		} catch (InterruptedException ex) {
			process.destroyForcibly();
			throw new InterruptedException("Mamba process stopped. The command being executed was: " + cmd);
		}
		// Wait for all output to be read
		outputThread.join();
		if (processResult != 0)
        	throw new RuntimeException("Error executing the following command: " + builder.command()
        								+ System.lineSeparator() + this.mambaConsoleOut
        								+ System.lineSeparator() + this.mambaConsoleErr);
	}
	
	private static boolean inputStreamOpen(InputStream inputStream) {

        try {
        	return inputStream.available() > 0;
        } catch (Exception ex) {
        	return false;
        }
	}

	/**
	 * Run a Conda command with one or more arguments.
	 * 
	 * @param args
	 *            One or more arguments for the Conda command.
	 * @throws RuntimeException
	 *             If there is any error running the commands
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public void runMamba(final String... args ) throws RuntimeException, IOException, InterruptedException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		runMamba(false, args);
	}

	/**
	 * Returns environment variables associated with the activated environment as
	 * {@code Map< String, String >}.
	 * 
	 * @return The environment variables as {@code Map< String, String >}.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 */
	/* TODO find equivalent in mamba
	public Map< String, String > getEnvironmentVariables() throws IOException, InterruptedException
	{
		return getEnvironmentVariables( envName );
	}
	*/

	/**
	 * Returns environment variables associated with the specified environment as
	 * {@code Map< String, String >}.
	 * 
	 * @param envName
	 *            The environment name used to run the Python command.
	 * @return The environment variables as {@code Map< String, String >}.
	 * @throws IOException
	 *             If an I/O error occurs.
	 * @throws InterruptedException
	 *             If the current thread is interrupted by another thread while it
	 *             is waiting, then the wait is ended and an InterruptedException is
	 *             thrown.
	 */
	/**
	 * TODO find equivalent in mamba
	public Map< String, String > getEnvironmentVariables( final String envName ) throws IOException, InterruptedException
	{
		final List< String > cmd = getBaseCommand();
		cmd.addAll( Arrays.asList( condaCommand, "env", "config", "vars", "list", "-n", envName ) );
		final Process process = getBuilder( false ).command( cmd ).start();
		if ( process.waitFor() != 0 )
			throw new RuntimeException();
		final Map< String, String > map = new HashMap<>();
		try (final BufferedReader reader = new BufferedReader( new InputStreamReader( process.getInputStream() ) ))
		{
			String line;

			while ( ( line = reader.readLine() ) != null )
			{
				final String[] keyVal = line.split( " = " );
				map.put( keyVal[ 0 ], keyVal[ 1 ] );
			}
		}
		return map;
	}
	*/

	/**
	 * Returns a list of the Mamba environment names as {@code List< String >}.
	 * 
	 * @return The list of the Mamba environment names as {@code List< String >}.
	 * @throws IOException If an I/O error occurs.
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public List< String > getEnvironmentNames() throws IOException, MambaInstallException
	{
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		final List< String > envs = new ArrayList<>( Arrays.asList( DEFAULT_ENVIRONMENT_NAME ) );
		envs.addAll( Files.list( Paths.get( envsdir ) )
				.map( p -> p.getFileName().toString() )
				.filter( p -> !p.startsWith( "." ) )
				.collect( Collectors.toList() ) );
		return envs;
	}
	
	/**
	 * Check whether a list of dependencies provided is installed in the wanted environment.
	 * 
	 * @param envName
	 * 	The name of the environment of interest. Should be one of the environments of the current Mamba instance.
	 * 	This parameter can also be the full path to an independent environment.
	 * @param dependencies
	 * 	The list of dependencies that should be installed in the environment.
	 * 	They can contain version requirements. The names should be the ones used to import the package inside python,
	 * 	"skimage", not "scikit-image" or "sklearn", not "scikit-learn"
	 * 	An example list: "numpy", "numba&gt;=0.43.1", "torch==1.6", "torch&gt;=1.6, &lt;2.0"
	 * @return true if the packages are installed or false otherwise
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public boolean checkAllDependenciesInEnv(String envName, List<String> dependencies) throws MambaInstallException {
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		return checkUninstalledDependenciesInEnv(envName, dependencies).size() == 0;
	}
	
	/**
	 * Returns a list containing the packages that are not installed in the wanted environment
	 * from the list of dependencies provided
	 * 
	 * @param envName
	 * 	The name of the environment of interest. Should be one of the environments of the current Mamba instance.
	 * 	This parameter can also be the full path to an independent environment.
	 * @param dependencies
	 * 	The list of dependencies that should be installed in the environment.
	 * 	They can contain version requirements. The names should be the ones used to import the package inside python,
	 * 	"skimage", not "scikit-image" or "sklearn", not "scikit-learn"
	 * 	An example list: "numpy", "numba&gt;=0.43.1", "torch==1.6", "torch&gt;=1.6, &lt;2.0"
	 * @return true if the packages are installed or false otherwise
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public List<String>  checkUninstalledDependenciesInEnv(String envName, List<String> dependencies) throws MambaInstallException {
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		File envFile = new File(this.envsdir, envName);
		File envFile2 = new File(envName);
		if (!envFile.isDirectory() && !envFile2.isDirectory())
			return dependencies;
		List<String> uninstalled = dependencies.stream().filter(dep -> {
			try {
				return !checkDependencyInEnv(envName, dep);
			} catch (Exception ex) {
				return true;
			}
		}).collect(Collectors.toList());
		System.out.println(uninstalled);
		return uninstalled;
	}
	
	/**
	 * Checks whether a package is installed in the wanted environment.
	 * TODO improve the logic for bigger or smaller versions
	 * 
	 * @param envName
	 * 	The name of the environment of interest. Should be one of the environments of the current Mamba instance.
	 * 	This parameter can also be the full path to an independent environment.
	 * @param dependency
	 * 	The name of the package that should be installed in the env
	 * 	They can contain version requirements. The names should be the ones used to import the package inside python,
	 * 	"skimage", not "scikit-image" or "sklearn", not "scikit-learn"
	 * 	An example list: "numpy", "numba&gt;=0.43.1", "torch==1.6", "torch&gt;=1.6, &lt;2.0"
	 * @return true if the package is installed or false otherwise
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public boolean checkDependencyInEnv(String envName, String dependency) throws MambaInstallException {
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		if (dependency.contains("=<"))
			throw new IllegalArgumentException("=< is not valid, use <=");
		else if (dependency.contains("=>"))
			throw new IllegalArgumentException("=> is not valid, use >=");
		else if (dependency.contains(">") && dependency.contains("<") && !dependency.contains(","))
			throw new IllegalArgumentException("Invalid dependency format. To specify both a minimum and maximum version, "
					+ "separate the conditions with a comma. For example: 'torch>2.0.0, torch<2.5.0'.");
		
		if (dependency.contains("==")) {
			int ind = dependency.indexOf("==");
			return checkDependencyInEnv(envName, dependency.substring(0, ind).trim(), dependency.substring(ind + 2).trim());
		} else if (dependency.contains(">=") && dependency.contains("<=") && dependency.contains(",")) {
			int commaInd = dependency.indexOf(",");
			int highInd = dependency.indexOf(">=");
			int lowInd = dependency.indexOf("<=");
			int minInd = Math.min(Math.min(commaInd, lowInd), highInd);
			String packName = dependency.substring(0, minInd).trim();
			String maxV = dependency.substring(lowInd + 2, lowInd < highInd ? commaInd : dependency.length());
			String minV = dependency.substring(highInd + 2, lowInd < highInd ? dependency.length() : commaInd);
			if (maxV.equals("") || minV.equals(""))
				throw new IllegalArgumentException("Conditions must always begin with either '<' or '>' signs and then "
						+ "the version number. For example: 'torch>=2.0.0, torch<=2.5.0'.");
			return checkDependencyInEnv(envName, packName, minV, maxV, false);
		} else if (dependency.contains(">=") && dependency.contains("<") && dependency.contains(",")) {
			int commaInd = dependency.indexOf(",");
			int highInd = dependency.indexOf(">=");
			int lowInd = dependency.indexOf("<");
			int minInd = Math.min(Math.min(commaInd, lowInd), highInd);
			String packName = dependency.substring(0, minInd).trim();
			String maxV = dependency.substring(lowInd + 1, lowInd < highInd ? commaInd : dependency.length());
			String minV = dependency.substring(highInd + 2, lowInd < highInd ? dependency.length() : commaInd);
			if (maxV.equals("") || minV.equals(""))
				throw new IllegalArgumentException("Conditions must always begin with either '<' or '>' signs and then "
						+ "the version number. For example: 'torch>=2.0.0, torch<2.5.0'.");
			return checkDependencyInEnv(envName, packName, minV, null, false) && checkDependencyInEnv(envName, packName, null, maxV, true);
		} else if (dependency.contains(">") && dependency.contains("<=") && dependency.contains(",")) {
			int commaInd = dependency.indexOf(",");
			int highInd = dependency.indexOf(">");
			int lowInd = dependency.indexOf("<=");
			int minInd = Math.min(Math.min(commaInd, lowInd), highInd);
			String packName = dependency.substring(0, minInd).trim();
			String maxV = dependency.substring(lowInd + 2, lowInd < highInd ? commaInd : dependency.length());
			String minV = dependency.substring(highInd + 1, lowInd < highInd ? dependency.length() : commaInd);
			if (maxV.equals("") || minV.equals(""))
				throw new IllegalArgumentException("Conditions must always begin with either '<' or '>' signs and then "
						+ "the version number. For example: 'torch>2.0.0, torch<=2.5.0'.");
			return checkDependencyInEnv(envName, packName, minV, null, true) && checkDependencyInEnv(envName, packName, null, maxV, false);
		} else if (dependency.contains(">") && dependency.contains("<") && dependency.contains(",")) {
			int commaInd = dependency.indexOf(",");
			int highInd = dependency.indexOf(">");
			int lowInd = dependency.indexOf("<");
			int minInd = Math.min(Math.min(commaInd, lowInd), highInd);
			String packName = dependency.substring(0, minInd).trim();
			String maxV = dependency.substring(lowInd + 1, lowInd < highInd ? commaInd : dependency.length());
			String minV = dependency.substring(highInd + 1, lowInd < highInd ? dependency.length() : commaInd);
			if (maxV.equals("") || minV.equals(""))
				throw new IllegalArgumentException("Conditions must always begin with either '<' or '>' signs and then "
						+ "the version number. For example: 'torch>2.0.0, torch<2.5.0'.");
			return checkDependencyInEnv(envName, packName, minV, maxV, true);
		} else if (dependency.contains(">=")) {
			int ind = dependency.indexOf(">=");
			String maxV = dependency.substring(ind + 2).trim();
			if (maxV.equals(""))
				throw new IllegalArgumentException("Conditions must always begin with either '<' or '>' signs and then "
						+ "the version number. For example: 'torch>=2.0.0'.");
			return checkDependencyInEnv(envName, dependency.substring(0, ind).trim(), maxV, null, false);
		} else if (dependency.contains(">")) {
			int ind = dependency.indexOf(">");
			String maxV = dependency.substring(ind + 1).trim();
			if (maxV.equals(""))
				throw new IllegalArgumentException("Conditions must always begin with either '<' or '>' signs and then "
						+ "the version number. For example: 'torch>2.0.0'.");
			return checkDependencyInEnv(envName, dependency.substring(0, ind).trim(), maxV, null, true);
		} else if (dependency.contains("<=")) {
			int ind = dependency.indexOf("<=");
			String maxV = dependency.substring(ind + 2).trim();
			if (maxV.equals(""))
				throw new IllegalArgumentException("Conditions must always begin with either '<' or '>' signs and then "
						+ "the version number. For example: 'torch<=2.0.0'.");
			return checkDependencyInEnv(envName, dependency.substring(0, ind).trim(), null, maxV, false);
		} else if (dependency.contains("<")) {
			int ind = dependency.indexOf("<");
			String maxV = dependency.substring(ind + 1).trim();
			if (maxV.equals(""))
				throw new IllegalArgumentException("Conditions must always begin with either '<' or '>' signs and then "
						+ "the version number. For example: 'torch<2.0.0'.");
			return checkDependencyInEnv(envName, dependency.substring(0, ind).trim(), null, maxV, true);
		} else if (dependency.contains("=")) {
			int ind = dependency.indexOf("=");
			return checkDependencyInEnv(envName, dependency.substring(0, ind).trim(), dependency.substring(ind + 1).trim());
		}else {
			return checkDependencyInEnv(envName, dependency, null);
		}
	}
	
	/**
	 * Checks whether a package of a specific version is installed in the wanted environment.
	 * 
	 * @param envDir
	 * 	The directory of the environment of interest. Should be one of the environments of the current Mamba instance.
	 * 	This parameter can also be the full path to an independent environment.
	 * @param dependency
	 * 	The name of the package that should be installed in the env. The String should only contain the name, no version,
	 * 	and the name should be the one used to import the package inside python. For example, "skimage", not "scikit-image"
	 *  or "sklearn", not "scikit-learn".
	 * @param version
	 * 	the specific version of the package that needs to be installed. For example:, "0.43.1", "1.6", "2.0"
	 * @return true if the package is installed or false otherwise
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public boolean checkDependencyInEnv(String envDir, String dependency, String version) throws MambaInstallException {
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		return checkDependencyInEnv(envDir, dependency, version, version, true);
	}
	
	/**
	 * Checks whether a package with specific version constraints is installed in the wanted environment.
	 * In this method the minversion argument should be strictly smaller than the version of interest and
	 * the maxversion strictly bigger.
	 * This method checks that: dependency &gt;minversion, &lt;maxversion
	 * For smaller or equal or bigger or equal (dependency &gt;=minversion, &lt;=maxversion) look at the method
	 * {@link #checkDependencyInEnv(String, String, String, String, boolean)} with the lst parameter set to false.
	 * 
	 * @param envDir
	 * 	The directory of the environment of interest. Should be one of the environments of the current Mamba instance.
	 * 	This parameter can also be the full path to an independent environment.
	 * @param dependency
	 * 	The name of the package that should be installed in the env. The String should only contain the name, no version,
	 * 	and the name should be the one used to import the package inside python. For example, "skimage", not "scikit-image"
	 *  or "sklearn", not "scikit-learn".
	 * @param minversion
	 * 	the minimum required version of the package that needs to be installed. For example:, "0.43.1", "1.6", "2.0".
	 * 	This version should be strictly smaller than the one of interest, if for example "1.9" is given, it is assumed that
	 * 	package_version&gt;1.9.
	 * 	If there is no minimum version requirement for the package of interest, set this argument to null.
	 * @param maxversion
	 * 	the maximum required version of the package that needs to be installed. For example:, "0.43.1", "1.6", "2.0".
	 * 	This version should be strictly bigger than the one of interest, if for example "1.9" is given, it is assumed that
	 * 	package_version&lt;1.9.
	 * 	If there is no maximum version requirement for the package of interest, set this argument to null.
	 * @return true if the package is installed or false otherwise
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public boolean checkDependencyInEnv(String envDir, String dependency, String minversion, String maxversion) throws MambaInstallException {
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		return checkDependencyInEnv(envDir, dependency, minversion, maxversion, true);
	}
	
	/**
	 * Checks whether a package with specific version constraints is installed in the wanted environment.
	 * Depending on the last argument ('strictlyBiggerOrSmaller') 'minversion' and 'maxversion'
	 * will be strictly bigger(&gt;=) or smaller(&lt;) or bigger or equal &gt;=) or smaller or equal&lt;=)
	 * In this method the minversion argument should be strictly smaller than the version of interest and
	 * the maxversion strictly bigger.
	 * 
	 * @param envDir
	 * 	The directory of the environment of interest. Should be one of the environments of the current Mamba instance.
	 * 	This parameter can also be the full path to an independent environment.
	 * @param dependency
	 * 	The name of the package that should be installed in the env. The String should only contain the name, no version,
	 * 	and the name should be the one used to import the package inside python. For example, "skimage", not "scikit-image"
	 *  or "sklearn", not "scikit-learn".
	 * @param minversion
	 * 	the minimum required version of the package that needs to be installed. For example:, "0.43.1", "1.6", "2.0".
	 * 	If there is no minimum version requirement for the package of interest, set this argument to null.
	 * @param maxversion
	 * 	the maximum required version of the package that needs to be installed. For example:, "0.43.1", "1.6", "2.0".
	 * 	If there is no maximum version requirement for the package of interest, set this argument to null.
	 * @param strictlyBiggerOrSmaller
	 * 	Whether the minversion and maxversion shuld be strictly smaller and bigger or not
	 * @return true if the package is installed or false otherwise
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public boolean checkDependencyInEnv(String envDir, String dependency, String minversion, 
			String maxversion, boolean strictlyBiggerOrSmaller) throws MambaInstallException {
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		File envFile = new File(this.envsdir, envDir);
		File envFile2 = new File(envDir);
		if (!envFile.isDirectory() && !envFile2.isDirectory())
			return false;
		else if (!envFile.isDirectory())
			envFile = envFile2;
		if (dependency.trim().equals("python")) return checkPythonInstallation(envDir, minversion, maxversion, strictlyBiggerOrSmaller);
		String checkDepCode;
		if (minversion != null && maxversion != null && minversion.equals(maxversion)) {
			checkDepCode = "import importlib.util, sys; "
					+ "from importlib.metadata import version; "
					+ "from packaging import version as vv; "
					+ "pkg = '%s'; wanted_v = '%s'; "
					+ "spec = importlib.util.find_spec(pkg); "
					+ "vv_og = vv.parse(version('%s')).base_version; "
					+ "vv_nw = vv.parse(wanted_v).base_version; "
					+ "sys.exit(1) if spec is None else None; "
					+ "sys.exit(1) if vv_og != vv_nw else None; "
					+ "sys.exit(0);";
			checkDepCode = String.format(checkDepCode, resolveAliases(dependency), maxversion, dependency);
		} else if (minversion == null && maxversion == null) {
			checkDepCode = "import importlib.util, sys; sys.exit(0) if importlib.util.find_spec('%s') else sys.exit(1)";
			checkDepCode = String.format(checkDepCode, resolveAliases(dependency));
		} else if (maxversion == null) {
			checkDepCode = "import importlib.util, sys; "
					+ "from importlib.metadata import version; "
					+ "from packaging import version as vv; "
					+ "pkg = '%s'; desired_version = '%s'; "
					+ "spec = importlib.util.find_spec(pkg); "
					+ "curr_v = vv.parse(version('%s')).base_version; "
					+ "sys.exit(0) if spec and curr_v %s vv.parse(desired_version).base_version else sys.exit(1)";
			checkDepCode = String.format(checkDepCode, resolveAliases(dependency), minversion, dependency, strictlyBiggerOrSmaller ? ">" : ">=");
		} else if (minversion == null) {
			checkDepCode = "import importlib.util, sys; "
					+ "from importlib.metadata import version; "
					+ "from packaging import version as vv; "
					+ "pkg = '%s'; desired_version = '%s'; "
					+ "spec = importlib.util.find_spec(pkg); "
					+ "curr_v = vv.parse(version('%s')).base_version; "
					+ "sys.exit(0) if spec and curr_v %s vv.parse(desired_version).base_version else sys.exit(1)";
			checkDepCode = String.format(checkDepCode, resolveAliases(dependency), maxversion, dependency, strictlyBiggerOrSmaller ? "<" : "<=");
		} else {
			checkDepCode = "import importlib.util, sys; "
					+ "from importlib.metadata import version; "
					+ "from packaging import version as vv; "
					+ "pkg = '%s'; min_v = '%s'; max_v = '%s'; "
					+ "spec = importlib.util.find_spec(pkg); "
					+ "curr_v = vv.parse(version('%s')).base_version; "
					+ "sys.exit(0) if spec and curr_v %s vv.parse(min_v).base_version and curr_v %s vv.parse(max_v).base_version else sys.exit(1)";
			checkDepCode = String.format(checkDepCode, resolveAliases(dependency), minversion, maxversion, dependency,
					strictlyBiggerOrSmaller ? ">" : ">=", strictlyBiggerOrSmaller ? "<" : "<=");
		}
		try {
			runPythonIn(envFile, "-c", checkDepCode);
		} catch (RuntimeException | IOException | InterruptedException e) {
			return false;
		}
		return true;
	}
	
	private static String resolveAliases(String dep) {
		if (dep.equals("pytorch"))
			return "torch";
		else if (dep.equals("opencv-python"))
			return "cv2";
		else if (dep.equals("scikit-image"))
			return "skimage";
		else if (dep.equals("scikit-learn"))
			return "sklearn";
		return dep.replace("-", "_");
	}
	
	private boolean checkPythonInstallation(String envDir, String minversion, String maxversion, boolean strictlyBiggerOrSmaller) throws MambaInstallException {
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		File envFile = new File(this.envsdir, envDir);
		File envFile2 = new File(envDir);
		if (!envFile.isDirectory() && !envFile2.isDirectory())
			return false;
		else if (!envFile.isDirectory())
			envFile = envFile2;
		String checkDepCode;
		if (minversion != null && maxversion != null && minversion.equals(maxversion)) {
			checkDepCode = "import sys; import platform; from packaging import version as vv; desired_version = '%s'; "
					+ "sys.exit(0) if vv.parse(platform.python_version()).base_version == vv.parse(desired_version).base_version"
					+ " else sys.exit(1)";
			checkDepCode = String.format(checkDepCode, maxversion);
		} else if (minversion == null && maxversion == null) {
			checkDepCode = "2 + 2";
		} else if (maxversion == null) {
			checkDepCode = "import sys; import platform; from packaging import version as vv; desired_version = '%s'; "
					+ "sys.exit(0) if "
					+ "vv.parse(platform.python_version()).base_version %s vv.parse(desired_version).base_version "
					+ "else sys.exit(1)";
			checkDepCode = String.format(checkDepCode, minversion, strictlyBiggerOrSmaller ? ">" : ">=");
		} else if (minversion == null) {
			checkDepCode = "import sys; import platform; from packaging import version as vv; desired_version = '%s'; "
					+ "sys.exit(0) if "
					+ "vv.parse(platform.python_version()).base_version %s vv.parse(desired_version).base_version "
					+ "else sys.exit(1)";
			checkDepCode = String.format(checkDepCode, maxversion, strictlyBiggerOrSmaller ? "<" : "<=");
		} else {
			checkDepCode = "import platform; "
					+ "from packaging import version as vv; min_v = '%s'; max_v = '%s'; "
					+ "sys.exit(0) if "
					+ "vv.parse(platform.python_version()).base_version %s vv.parse(min_v).base_version "
					+ "and vv.parse(platform.python_version()).base_version %s vv.parse(max_v).base_version "
					+ "else sys.exit(1)";
			checkDepCode = String.format(checkDepCode, minversion, maxversion, strictlyBiggerOrSmaller ? ">" : ">=", strictlyBiggerOrSmaller ? "<" : ">=");
		}
		try {
			runPythonIn(envFile, "-c", checkDepCode);
		} catch (RuntimeException | IOException | InterruptedException e) {
			return false;
		}
		return true;
	}
	
	/**
	 * TODO figure out whether to use a dependency or not to parse the yaml file
	 * @param envYaml
	 * 	the path to the yaml file where a Python environment should be specified
	 * @return true if the env exists or false otherwise
	 * @throws MambaInstallException if Micromamba has not been installed, thus the instance of {@link Mamba} cannot be used
	 */
	public boolean checkEnvFromYamlExists(String envYaml) throws MambaInstallException {
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		if (envYaml == null || new File(envYaml).isFile() == false 
				|| (envYaml.endsWith(".yaml") && envYaml.endsWith(".yml"))) {
			return false;
		}
		return false;
	}
	
	/**
	 * In Windows, if a command prompt argument contains and space " " it needs to
	 * start and end with double quotes
	 * @param arg
	 * 	the cmd argument
	 * @return a robust argument
	 */
	private static String coverArgWithDoubleQuotes(String arg) {
		String[] specialChars = new String[] {" "};
        for (String schar : specialChars) {
        	if (arg.startsWith("\"") && arg.endsWith("\""))
        		continue;
        	if (arg.contains(schar) && PlatformDetection.isWindows()) {
        		return "\"" + arg + "\"";
        	}
        }
        return arg;
	}
	
	/**
	 * When an argument of a command prompt argument in Windows contains an space, not
	 * only the argument needs to be surrounded by double quotes, but the whole sentence
	 * @param args
	 * 	arguments to be executed by the windows cmd
	 * @return a complete Sting containing all the arguments and surrounded by double quotes
	 */
	private static String surroundWithQuotes(List<String> args) {
		String arg = "\"";
		for (String aa : args) {
			arg += aa + " ";
		}
		arg = arg.substring(0, arg.length() - 1);
		arg += "\"";
		return arg;
	}
	
	public static void main(String[] args) throws IOException, InterruptedException, MambaInstallException {
		
		Mamba m = new Mamba("/home/carlos/.local/share/appose/micromamba");
		boolean aa = m.checkDependencyInEnv("biapy", "scikit-learn>=1.4.0");
		System.out.println(aa);
	}
	
	/**
	 * TODO keep until release of stable Appose
	 * Install the Python package to run Appose in Python
	 * @param envName
	 * 	environment where Appose is going to be installed
	 * @throws IOException if there is any file creation related issue
	 * @throws InterruptedException if the package installation is interrupted
	 * @throws MambaInstallException if there is any error with the Mamba installation
	 */
	private void installApposeFromSource(String envName) throws IOException, InterruptedException, MambaInstallException {
		checkMambaInstalled();
		if (!installed) throw new MambaInstallException("Micromamba is not installed");
		String zipResourcePath = "appose-python.zip";
        String outputDirectory = this.getEnvsDir() + File.separator + envName;
		if (new File(envName).isDirectory()) outputDirectory = new File(envName).getAbsolutePath();
        try (
            	InputStream zipInputStream = Mamba.class.getClassLoader().getResourceAsStream(zipResourcePath);
            	ZipInputStream zipInput = new ZipInputStream(zipInputStream);
            		) {
            	ZipEntry entry;
            	while ((entry = zipInput.getNextEntry()) != null) {
                    File entryFile = new File(outputDirectory + File.separator + entry.getName());
                    if (entry.isDirectory()) {
                    	entryFile.mkdirs();
                    	continue;
                    }
                	entryFile.getParentFile().mkdirs();
                    try (OutputStream entryOutput = new FileOutputStream(entryFile)) {
                        byte[] buffer = new byte[1024];
                        int bytesRead;
                        while ((bytesRead = zipInput.read(buffer)) != -1) {
                            entryOutput.write(buffer, 0, bytesRead);
                        }
                    }
                }
            }
        this.pipInstallIn(envName, new String[] {outputDirectory + File.separator + "appose-python"});
	}

}
