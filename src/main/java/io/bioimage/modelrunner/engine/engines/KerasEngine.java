package io.bioimage.modelrunner.engine.engines;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.engine.AbstractEngine;

public class KerasEngine extends AbstractEngine {

	private static final List<String> SUPPORTED_KERAS_VERSIONS = Arrays.stream(new String[] {}).collect(Collectors.toList());
	private static final List<String> SUPPORTED_KERAS_VERSION_NUMBERS = Arrays.stream(new String[] {}).collect(Collectors.toList());
	
	private KerasEngine(String version, Boolean cpu, Boolean gpu, Boolean isPython) {
		if (!isPython) throw new IllegalArgumentException("JDLL only has support for Keras through a Python engine.");
		Mamba mamba = new Mamba();
		mamba.getEnvsDir()
	}

	
	public static KerasEngine initialize(String version, Boolean cpu, Boolean gpu, Boolean isPython) {
		return new KerasEngine(version, cpu, gpu, isPython);
	}
	
	public static String getDir(String version, boolean cpu, boolean gpu, boolean isPython) {
		if (!isPython) 
			throw new IllegalArgumentException("JDLL only has support for Keras through a Python engine.");
		if (!SUPPORTED_KERAS_VERSION_NUMBERS.contains(version))
			throw new IllegalArgumentException("The provided Keras version is not supported by JDLL: " + version
					+ ". The supported versions are: " + SUPPORTED_KERAS_VERSION_NUMBERS);
		if (!SUPPORTED_KERAS_VERSION_NUMBERS.contains(version))
	}
	
	@Override
	public String getName() {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public String getDir() {
		// TODO Auto-generated method stub
		return null;
	}

}
