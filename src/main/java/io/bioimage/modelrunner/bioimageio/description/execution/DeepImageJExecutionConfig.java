package io.bioimage.modelrunner.bioimageio.description.execution;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Execution configuration Information for DeepImageJ.
 * 
 * @author Daniel Felipe Gonzalez Obando
 */
public class DeepImageJExecutionConfig
{
	/**
	 * Pre-processing key in the rdf.yaml DIJ config
	 */
	private static String preKey = "preprocess";
	/**
	 * Post-processing key in the rdf.yaml DIJ config
	 */
	private static String postKey = "postprocess";
	/**
	 * Prediction key in the DIJ config
	 */
	private static String predictionKey = "prediction";
	/**
	 * Pyramidal model key in the DIJ config
	 */
	private static String pyramidKey = "pyramidal_model";
	/**
	 * Tiling key in the DIJ config
	 */
	private static String tilingKey = "allow_tiling";
	/**
	 * Test inforamtion key in the DIJ config
	 */
	private static String testInfoKey = "test_information";

    /**
     * Creates an execution configuration information instance from the element map.
     * 
     * @param yamlMap
     *        The element map.
     * @return The execution configuration instance.
     */
    public static DeepImageJExecutionConfig build(Map<String, Object> yamlMap)
    {
        DeepImageJExecutionConfig config = new DeepImageJExecutionConfig();

        config.pyramidalModel = yamlMap.containsKey(pyramidKey)
            ? (boolean) yamlMap.get(pyramidKey)
            : false;
        config.allowTiling = yamlMap.containsKey(tilingKey) ? (boolean) yamlMap.get(tilingKey) : false;

        @SuppressWarnings({"unchecked", "unused"})
        Map<String, Object> testInformationMap = yamlMap.containsKey(testInfoKey)
            ? ((Map<String, Object>) yamlMap.get(testInfoKey))
            : Collections.EMPTY_MAP;

        @SuppressWarnings({"unchecked"})
        Map<String, Object> predictionMap = ((Map<String, Object>) yamlMap.get(predictionKey));
        config.createProcessings(predictionMap);

        return config;
    }
    
    /**
     * Create the pre- and post-processing for the DIJ config in the rdf.yaml
     * @param predictionMap
     * 	the prediction part in the DIJ config in the rdf.yaml
     */
    private void createProcessings(Object predictionMap) {
    	if (predictionMap == null || !(predictionMap instanceof HashMap<?, ?>)
    			|| !(predictionMap instanceof Map<?, ?>)) {
    		pre = DeepImageJProcessing.build(null);
    		post = DeepImageJProcessing.build(null);
    		return;
    	}
    	if (predictionMap instanceof HashMap<?, ?>) {
    		pre = DeepImageJProcessing.build(((HashMap<?, ?>)predictionMap).get(preKey));
    		post = DeepImageJProcessing.build(((HashMap<?, ?>)predictionMap).get(postKey));
    	} else if (predictionMap instanceof Map<?, ?>) {
    		pre = DeepImageJProcessing.build(((Map<?, ?>)predictionMap).get(preKey));
    		post = DeepImageJProcessing.build(((Map<?, ?>)predictionMap).get(postKey));
    	}
    }

    private boolean pyramidalModel = false;
    /**
     * Tiling is always allowed unless the contrary is specified
     */
    private boolean allowTiling = true;
    private String tensorFlowModelTag;
    private String tensorFlowSignatureDef;
    private DeepImageJProcessing pre;
    private DeepImageJProcessing post;

    private DeepImageJExecutionConfig()
    {
    }

    /**
     * @return True if this model is pyramidal.
     */
    public boolean isPyramidalModel()
    {
        return pyramidalModel;
    }

    /**
     * @return True if this model allows tensor tiling.
     */
    public boolean isAllowTiling()
    {
        return allowTiling;
    }

    /**
     * @return The tag used to load the model in memory.
     */
    public String getTensorFlowModelTag()
    {
        return tensorFlowModelTag;
    }

    /**
     * @return The name of the signature in the loaded TensorFlow model for this specific model.
     */
    public String getTensorFlowSignatureDef()
    {
        return tensorFlowSignatureDef;
    }
    
    /**
     * 
     * @return the DeepImageJ pre-processing specified in the rdf.yaml
     */
    public DeepImageJProcessing getPreprocesing() {
    	return pre;
    }
    
    /**
     * 
     * @return the DeepImageJ post-processing specified in the rdf.yaml
     */
    public DeepImageJProcessing getPostprocesing() {
    	return post;
    }
}
