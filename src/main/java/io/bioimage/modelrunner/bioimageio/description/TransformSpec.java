package io.bioimage.modelrunner.bioimageio.description;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import io.bioimage.modelrunner.bioimageio.description.execution.DeepImageJProcessing;

/**
 * A pre or post processing transformation descriptor applied on a tensor.
 * 
 * @author Daniel Felipe Gonzalez Obando and Carlos Garcia Lopez de Haro
 */
public class TransformSpec
{
	/**
	 * Whether the transformation represents a DIJ transformation or not
	 */
	private boolean isDIJ = false;
	private static String kwargsKey = "kwargs";
	private static String transformationNameKey = "name";
	
    /**
     * Builds a transformation specification from the provided element map.
     * 
     * @param transformSpecMap
     *        Element map.
     * @return The transformation specification instance.
     */
    public static TransformSpec build(Map<String, Object> transformSpecMap)
    {
        TransformSpec transform = new TransformSpec();
        transform.specMap = transformSpecMap;
        return transform;
    }

    private Map<String, Object> specMap;

    /**
     * @return The specification map describing the transformation and the parameters used.
     */
    public Map<String, Object> getSpecMap()
    {
        return specMap == null ? null : Collections.unmodifiableMap(specMap);
    }

    /**
     * @return The transformation name. Null if the specification map is not specified or if the transformation has no name.
     */
    public String getName()
    {
        return specMap == null ? null : (String) specMap.get("name");
    }

    /**
     * @return The keyword arguments for this transformation. Null if the specification map is not specified or if the transformation has no kwargs element.
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> getKwargs()
    {
        return specMap == null ? null : (Map<String, Object>) specMap.get("kwargs");
    }
    
    /**
     * 
     * @return whether the transformation represents a DIJ transformation
     */
    public boolean isDIJ() {
    	return isDIJ;
    }
    
    /**
     * Create a {@link TransformSpec} object from a DIJ transformation
     * @param dij
     * 	the DIj transformation
     * @return the DIJ trnasformation as {@link TransformSpec}
     */
    public static TransformSpec createTransformForDIJ(DeepImageJProcessing dij) {
    	Map<String, Object> transformSpecs = new HashMap<String, Object>();
    	transformSpecs.put("name", dij.getMacros().toString());
    	TransformSpec transform = new TransformSpec();
    	transform.specMap = transformSpecs;
    	transform.isDIJ = true;
    	return transform;
    }
    
    public static String getKwargsKey() {
    	return kwargsKey;
    }
    
    public static String getTransformationNameKey() {
    	return transformationNameKey;
    }

    @Override
    public String toString()
    {
        return "TransformSpec {specMap=" + specMap + "}";
    }
}
