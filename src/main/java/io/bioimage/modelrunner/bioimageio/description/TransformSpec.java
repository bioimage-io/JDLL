/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.bioimageio.description;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import io.bioimage.modelrunner.bioimageio.description.deepimagej.DeepImageJProcessing;

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
	private static String KWARGS_KEY = "kwargs";
	private static String TRANSFORMATION_NAME_KEY = "name";
	
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
        return specMap == null ? null : specMap;
    }

    /**
     * @return The transformation name. Null if the specification map is not specified or if the transformation has no name.
     */
    public String getName()
    {
    	if (specMap == null)
    		return null;
    	if (specMap.get("name") != null)
    		return (String) specMap.get("name");
    	return (String) specMap.get("id");
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
    	return KWARGS_KEY;
    }
    
    public static String getTransformationNameKey() {
    	return TRANSFORMATION_NAME_KEY;
    }

    @Override
    public String toString()
    {
        return "TransformSpec {specMap=" + specMap + "}";
    }
}
