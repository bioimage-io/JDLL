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
package io.bioimage.modelrunner.bioimageio.description;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;

import io.bioimage.modelrunner.transformations.PythonTransformation;

/**
 * Class to handle special mdoels such as cellpose or Stardist that are not fully defined by the rdf.yaml file
 * yet.
 * 
 * TODO this class will be removed or modified when the rdf.yaml file covers this models
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class SpecialModels
{
    
    protected static final String STARDIST_KEY = "stardist";
    
    protected static final String CELLPOSE_KEY = "stardist";
    
    /**
     * Method that checks whether the descriptor corresponds to a special model
     * or not and if it does belong to one of them , makes the pertinent modifications to 
     * be able to run it end to end
     * @param descriptor
     * 	the {@link ModelDescriptor} of the model of interest
     */
    public static void checkSpecialModels(ModelDescriptor descriptor)
    {
    	try {
    	if (descriptor.getConfig().getSpecMap().containsKey(STARDIST_KEY))
    		completeStardist(descriptor);
    	else if (descriptor.getConfig().getSpecMap().containsKey(CELLPOSE_KEY))
    		completeCellpose(descriptor);
    	} catch (Exception e) {
    		e.printStackTrace();
    	}
    }
    
    private static void completeStardist(ModelDescriptor descriptor) {
    	Map<String, Object> stardistMap = (Map<String, Object>) descriptor.getConfig().getSpecMap().get(STARDIST_KEY);
    	Map<String, Object> configMap = (Map<String, Object>) stardistMap.get("config");
    	Map<String, Object> stardistThres = (Map<String, Object>) stardistMap.get("thresholds");
    	Map<String, Object> stardistPostProcessing = new HashMap<String, Object>();
    	stardistPostProcessing.put(TransformSpec.getTransformationNameKey(), PythonTransformation.NAME);
    	stardistPostProcessing.put(PythonTransformation.ENV_YAML_KEY, "stardist.yaml");
    	stardistPostProcessing.put(PythonTransformation.SCRIPT_KEY, "stardist_postprocessing.py");
    	stardistPostProcessing.put(PythonTransformation.N_OUTPUTS_KEY, 1);
    	stardistPostProcessing.put(PythonTransformation.METHOD_KEY, "stardist_postprocessing");
    	Map<String, Object> kwargs = new HashMap<String, Object>();
    	kwargs.put("nms_thresh", stardistThres.get(kwargs));
    	kwargs.put("prob_thresh", stardistThres.get(kwargs));
    	stardistPostProcessing.put(PythonTransformation.KWARGS_KEY, kwargs);
    	if (extractStardist(descriptor))
        	descriptor.getOutputTensors().get(0).getPostprocessing().add(TransformSpec.build(stardistPostProcessing));
    }
    
    private static boolean extractStardist(ModelDescriptor descriptor) {
    	if (descriptor.getModelPath() == null) {
    		return true;
    	}
        File envFile = new File(descriptor.getModelPath() + File.separator + "stardist.yaml");
    	File scriptFile = new File(descriptor.getModelPath() + File.separator + "stardist_postprocessing.py");
    	if (!envFile.isFile()) {
    		try (InputStream envStream = SpecialModels.class.getClassLoader()
        			.getResourceAsStream("op_environments" + File.separator + "stardist.yaml")){
    			Files.copy(envStream, envFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
    		} catch (IOException e) {
    			e.printStackTrace();
    			return false;
    		}
    	}
    	if (!scriptFile.isFile()) {
    		try (InputStream scriptStream = SpecialModels.class.getClassLoader()
        			.getResourceAsStream("ops" + File.separator + "stardist_postprocessing" + File.separator + "stardist_postprocessing.py")){
    			Files.copy(scriptStream, scriptFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
    		} catch (IOException e) {
    			e.printStackTrace();
    			return false;
    		}
    	}
    	return true;
    }
    
    private static void completeCellpose(ModelDescriptor descriptor) {
    	return;
    }
}
