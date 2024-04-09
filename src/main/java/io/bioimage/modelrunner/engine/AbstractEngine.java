package io.bioimage.modelrunner.engine;

import java.util.Arrays;

public class AbstractEngine {
	
	private static String JAX_KEY = "jax";
	
	private static String PYTORCH_STATE_DIC_KEY = "pytorch_state_dict";
	
	private static final String[] SUPPORTED_ENGINE_NAMES = new String[] {EngineInfo.getTensorflowKey(), EngineInfo.getBioimageioTfKey(),
			EngineInfo.getBioimageioPytorchKey(), EngineInfo.getPytorchKey(), EngineInfo.getOnnxKey(), EngineInfo.getKerasKey(),
			EngineInfo.getBioimageioKerasKey(), PYTORCH_STATE_DIC_KEY, JAX_KEY};
	
	
	public static AbstractEngine initialize(String name, String version, Boolean cpu, Boolean gpu, Boolean isPython) {
		if (!isSupported(name)) throw new IllegalArgumentException("Name provided is not on the list of supported engine keys: "
				+ Arrays.toString(SUPPORTED_ENGINE_NAMES));
		if (KerasEngine.NAME.equals(name)) {
			
		} else if (TensorflowEngine.NAME.equals(name)) {
			return TensorflowEngine.initilize();
		} else if (KerasEngine.NAME.equals(name)) {
			return KerasEngine.initilize();
		} else if (PytorchEngine.NAME.equals(name)) {
			return PytorchEngine.initilize();
		} else if (TorchscriptEngine.NAME.equals(name)) {
			return TorchscriptEngine.initilize();
		} else if (JaxEngine.NAME.equals(name)) {
			return JaxEngine.initilize();
		} else if (OnnxEngine.NAME.equals(name)) {
			return OnnxEngine.initilize();
		}
	}
	
	public static boolean isSupported(String name) {
		return Arrays.stream(SUPPORTED_ENGINE_NAMES).filter(kk -> name.equals(kk)).findFirst().orElse(null) != null;
	}
	
	public static String[] getSupportedEngineKeys() {
		return SUPPORTED_ENGINE_NAMES;
	}

}
