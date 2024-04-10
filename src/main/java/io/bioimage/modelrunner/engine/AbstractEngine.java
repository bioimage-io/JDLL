package io.bioimage.modelrunner.engine;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.compress.archivers.ArchiveException;

import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat;
import io.bioimage.modelrunner.engine.engines.JaxEngine;
import io.bioimage.modelrunner.engine.engines.KerasEngine;
import io.bioimage.modelrunner.engine.engines.OnnxEngine;
import io.bioimage.modelrunner.engine.engines.PytorchEngine;
import io.bioimage.modelrunner.engine.engines.TensorflowEngine;
import io.bioimage.modelrunner.engine.engines.TorchscriptEngine;
import io.bioimage.modelrunner.model.Model;

public abstract class AbstractEngine {
	
	private static String JAX_KEY = "jax";
	
	private static String PYTORCH_STATE_DIC_KEY = "pytorch_state_dict";
	
	private static final String[] SUPPORTED_ENGINE_NAMES = new String[] {EngineInfo.getTensorflowKey(), EngineInfo.getBioimageioTfKey(),
			EngineInfo.getBioimageioPytorchKey(), EngineInfo.getPytorchKey(), EngineInfo.getOnnxKey(), EngineInfo.getKerasKey(),
			EngineInfo.getBioimageioKerasKey(), PYTORCH_STATE_DIC_KEY, JAX_KEY};
	
	public static AbstractEngine initialize(WeightFormat ww) {
		return null;
	}
	
	
	public static AbstractEngine initialize(String name, String version, boolean gpu, boolean isPython) {
		if (!isSupported(name)) throw new IllegalArgumentException("Name provided is not on the list of supported engine keys: "
				+ Arrays.toString(SUPPORTED_ENGINE_NAMES));
		if (KerasEngine.NAME.equals(name)) {
			return KerasEngine.initilize(version, gpu, isPython);
		} else if (TensorflowEngine.NAME.equals(name)) {
			return TensorflowEngine.initilize(version, gpu, isPython);
		} else if (PytorchEngine.NAME.equals(name)) {
			return PytorchEngine.initilize(version, gpu, isPython);
		} else if (TorchscriptEngine.NAME.equals(name)) {
			return TorchscriptEngine.initilize(version, gpu, isPython);
		} else if (JaxEngine.NAME.equals(name)) {
			return JaxEngine.initilize(version, gpu, isPython);
		} else if (OnnxEngine.NAME.equals(name)) {
			return OnnxEngine.initilize(version, gpu, isPython);
		}
	}
	
	public static AbstractEngine initializeSimilar(WeightFormat ww) {
		return null;
	}
	
	
	public static AbstractEngine initializeSimilar(String name, String version, Boolean gpu, Boolean isPython) {
		if (!isSupported(name)) throw new IllegalArgumentException("Name provided is not on the list of supported engine keys: "
				+ Arrays.toString(SUPPORTED_ENGINE_NAMES));
		if (KerasEngine.NAME.equals(name)) {
			return KerasEngine.initilize(version, gpu, isPython);
		} else if (TensorflowEngine.NAME.equals(name)) {
			return TensorflowEngine.initilize(version, gpu, isPython);
		} else if (PytorchEngine.NAME.equals(name)) {
			return PytorchEngine.initilize(version, gpu, isPython);
		} else if (TorchscriptEngine.NAME.equals(name)) {
			return TorchscriptEngine.initilize(version, gpu, isPython);
		} else if (JaxEngine.NAME.equals(name)) {
			return JaxEngine.initilize(version, gpu, isPython);
		} else if (OnnxEngine.NAME.equals(name)) {
			return OnnxEngine.initilize(version, gpu, isPython);
		}
	}
	
	public static boolean isSupported(String name) {
		return Arrays.stream(SUPPORTED_ENGINE_NAMES).filter(kk -> name.equals(kk)).findFirst().orElse(null) != null;
	}
	
	public static String[] getSupportedEngineKeys() {
		return SUPPORTED_ENGINE_NAMES;
	}
	
	public abstract String getName();
	
	public abstract boolean isPython();
	
	public abstract String getVersion();
	
	public abstract boolean supportsGPU();

	public abstract String getDir();

	public abstract boolean isInstalled();

	public abstract void install() throws IOException, InterruptedException, MambaInstallException, ArchiveException, URISyntaxException;

	public abstract Model load(String modelFolder, String modelSource);
	
}
