package io.bioimage.modelrunner.engine.engines;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.compress.archivers.ArchiveException;

import io.bioimage.modelrunner.apposed.appose.Environment;
import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.apposed.appose.Service;
import io.bioimage.modelrunner.engine.AbstractEngine;
import io.bioimage.modelrunner.model.Model;

public class KerasEngine extends AbstractEngine {
	
	private Mamba mamba;
	
	private String version;
	
	private boolean gpu;
	
	private boolean isPython;
	
	private Boolean installed;

	private Environment env;
	
	private Service python;
	
	public static final String NAME = "keras";

	private static final List<String> SUPPORTED_KERAS_GPU_VERSIONS = Arrays.stream(new String[] {}).collect(Collectors.toList());
	private static final List<String> SUPPORTED_KERAS_VERSION_NUMBERS = Arrays.stream(new String[] {}).collect(Collectors.toList());
	
	private KerasEngine(String version, boolean gpu, boolean isPython) {
		if (!isPython) 
			throw new IllegalArgumentException("JDLL only has support for Keras through a Python engine.");
		if (!SUPPORTED_KERAS_VERSION_NUMBERS.contains(version))
			throw new IllegalArgumentException("The provided Keras version is not supported by JDLL: " + version
					+ ". The supported versions are: " + SUPPORTED_KERAS_VERSION_NUMBERS);
		if (gpu && !SUPPORTED_KERAS_GPU_VERSIONS.contains(version))
			throw new IllegalArgumentException("The provided Keras version has no GPU support in JDLL: " + version
					+ ". GPU supported versions are: " + SUPPORTED_KERAS_GPU_VERSIONS);
		mamba = new Mamba();
		this.isPython = isPython;
		this.version = version;
	}

	
	public static KerasEngine initialize(String version, boolean gpu, boolean isPython) {
		return new KerasEngine(version, gpu, isPython);
	}
	
	public static String getFolderName(String version, boolean gpu, boolean isPython) {
		if (!isPython) 
			throw new IllegalArgumentException("JDLL only has support for Keras through a Python engine.");
		if (!SUPPORTED_KERAS_VERSION_NUMBERS.contains(version))
			throw new IllegalArgumentException("The provided Keras version is not supported by JDLL: " + version
					+ ". The supported versions are: " + SUPPORTED_KERAS_VERSION_NUMBERS);
		if (gpu && !SUPPORTED_KERAS_GPU_VERSIONS.contains(version))
			throw new IllegalArgumentException("The provided Keras version has no GPU support in JDLL: " + version
					+ ". GPU supported versions are: " + SUPPORTED_KERAS_GPU_VERSIONS);
		return NAME + "_" + version + (gpu ? "_gpu" : "");
	}
	
	public static List<KerasEngine> getInstalledVersions() {
		List<KerasEngine> cpus = SUPPORTED_KERAS_VERSION_NUMBERS.stream()
				.map(str -> new KerasEngine(str, false, true))
				.filter(vv -> vv.isInstalled()).collect(Collectors.toList());
		List<KerasEngine> gpus = SUPPORTED_KERAS_VERSION_NUMBERS.stream()
				.map(str -> new KerasEngine(str, false, true))
				.filter(vv -> vv.isInstalled()).collect(Collectors.toList());
		cpus.addAll(gpus);
		return cpus;
	}
	
	@Override
	public String getName() {
		return NAME;
	}
	
	@Override
	public String getDir() {
		return mamba.getEnvsDir() + File.separator + getFolderName(version, gpu, false);
	}


	@Override
	public boolean isPython() {
		return isPython;
	}


	@Override
	public String getVersion() {
		return version;
	}


	@Override
	public boolean supportsGPU() {
		return gpu;
	}


	@Override
	public boolean isInstalled() {
		if (installed != null)
			return installed;
		if (!(new File(getDir()).exists()))
			return false;
		installed = getInstalledVersions().stream()
				.filter(vv -> vv.gpu == gpu && vv.version.equals(version)).findFirst().orElse(null) != null;
		return installed;
	}


	@Override
	public void install() throws IOException, InterruptedException, MambaInstallException, ArchiveException, URISyntaxException {
		if (!mamba.checkMambaInstalled()) mamba.installMicromamba();
		
		mamba.create(getDir(), getSupportedEngineKeys());
		installed = true;
	}


	@Override
	public Model load(String modelFolder, String modelSource) throws IOException {
		if (!this.isInstalled())
			throw new IllegalArgumentException("Current engine '" + this.toString() 
												+ "' is not installed. Please install it first.");

		this.env = new Environment() {
			@Override public String base() { return KerasEngine.this.getDir(); }
			@Override public boolean useSystemPath() { return false; }
			};
		python = env.python();
		return null;
	}
	
	@Override
	public String toString() {
		return NAME + "_" + version + (gpu ? "_gpu" : "");
	}

}
