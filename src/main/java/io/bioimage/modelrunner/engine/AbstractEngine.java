package io.bioimage.modelrunner.engine;

import java.util.Arrays;

public abstract class AbstractEngine {
	
	private static String JAX_KEY = "jax";
	
	private static String PYTORCH_STATE_DIC_KEY = "pytorch_state_dict";
	
	private static final String[] SUPPORTED_ENGINE_NAMES = new String[] {EngineInfo.getTensorflowKey(), EngineInfo.getBioimageioTfKey(),
			EngineInfo.getBioimageioPytorchKey(), EngineInfo.getPytorchKey(), EngineInfo.getOnnxKey(), EngineInfo.getKerasKey(),
			EngineInfo.getBioimageioKerasKey(), PYTORCH_STATE_DIC_KEY, JAX_KEY};
	
	
	public static AbstractEngine initialize(String name, String version, Boolean cpu, Boolean gpu, Boolean isPython) {
		if (!isSupported(name)) throw new IllegalArgumentException("Name provided is not on the list of supported engine keys: "
				+ Arrays.toString(SUPPORTED_ENGINE_NAMES));
		if (KerasEngine.NAME.equals(name)) {
			return KerasEngine.initilize(version, cpu, gpu, isPython);
		} else if (TensorflowEngine.NAME.equals(name)) {
			return TensorflowEngine.initilize(version, cpu, gpu, isPython);
		} else if (PytorchEngine.NAME.equals(name)) {
			return PytorchEngine.initilize(version, cpu, gpu, isPython);
		} else if (TorchscriptEngine.NAME.equals(name)) {
			return TorchscriptEngine.initilize(version, cpu, gpu, isPython);
		} else if (JaxEngine.NAME.equals(name)) {
			return JaxEngine.initilize(version, cpu, gpu, isPython);
		} else if (OnnxEngine.NAME.equals(name)) {
			return OnnxEngine.initilize(version, cpu, gpu, isPython);
		}
	}
	
	public static boolean isSupported(String name) {
		return Arrays.stream(SUPPORTED_ENGINE_NAMES).filter(kk -> name.equals(kk)).findFirst().orElse(null) != null;
	}
	
	public static String[] getSupportedEngineKeys() {
		return SUPPORTED_ENGINE_NAMES;
	}
	
	public abstract String getName();

}
