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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.description.cellpose.ModelDescriptorCellposeV04;
import io.bioimage.modelrunner.bioimageio.description.cellpose.ModelDescriptorCellposeV05;
import io.bioimage.modelrunner.bioimageio.description.stardist.ModelDescriptorStardistV04;
import io.bioimage.modelrunner.bioimageio.description.stardist.ModelDescriptorStardistV05;
import io.bioimage.modelrunner.model.cellpose.Cellpose;
import io.bioimage.modelrunner.model.stardist.StardistAbstract;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.utils.YAMLUtils;


/**
 * A data structure holding a single Bioimage.io pretrained model description. This instances are created by opening a {@code model.yaml} file.
 * More info about the parameters can be found at:
 * https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/model_spec_latest.md
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public class ModelDescriptorFactory {
	
	private static final String V05_START = "0.5";
	
	private static final String V04_START = "0.4";
	
	private static final String FORMAT = "format_version";
    
    /**
     * Opens the provided file and builds an instance of {@link ModelDescriptor} from it.
     * 
     * @param modelFile
     * 	Model descriptor file.
     * @return The instance of the model descriptor.
     * @throws IOException if any of the required files is incorrect or corrupted
     * @throws FileNotFoundException if any of the required files is missing
     */
    public static ModelDescriptor readFromLocalFile(String modelFile) throws FileNotFoundException, IOException
    {
    	Map<String, Object> yamlElements = YAMLUtils.load(modelFile);
        // TODO yamlElements.put(fromLocalKey, true);
    	// TODO yamlElements.put(modelPathKey, new File(modelFile).getParent());
    	yamlElements.put("modelPath", new File(modelFile).getParentFile().getAbsolutePath());
        return fromMap(yamlElements);
    }
    
    /**
     * Reads a yaml text String and builds an instance of {@link ModelDescriptorFactory} from it.
     * 
     * @param yamlText
     * 	text read from a yaml file that contains an rdf.yaml file
     * @return The instance of the model descriptor.
     */
    public static ModelDescriptor readFromYamlTextString(String yamlText)
    {
    	// Convert the String of text that contains the yaml file into Map
    	Map<String,Object> yamlElements = YAMLUtils.loadFromString(yamlText);
        return fromMap(yamlElements);
    }

    private static ModelDescriptor fromMap(Map<String,Object> yamlElements)
    {
    	if (isStardist(yamlElements) && StardistAbstract.isInstalled()) {
    		return fromStardistMap(yamlElements);
    	} else if (isCellpose(yamlElements) && Cellpose.isInstalled()) {
    		return fromCellposeMap(yamlElements);
    	}
    	Object formatVersion = yamlElements.get(FORMAT);
    	if (formatVersion instanceof String && ((String) formatVersion).startsWith(V04_START)) {
    		return new ModelDescriptorV04(yamlElements);
    	} else if (formatVersion instanceof String && ((String) formatVersion).startsWith(V05_START)) {
    		return new ModelDescriptorV05(yamlElements);
    	} else if (formatVersion instanceof String)
    		throw new IllegalArgumentException("JDLL only supports the Bioimage.io model specs 0.4 and 0.5.");
    	else {
    		throw new IllegalArgumentException("Incorrect format, missing 'format_version' field.");
    	}
    }

    private static ModelDescriptor fromStardistMap(Map<String,Object> yamlElements)
    {
    	Object formatVersion = yamlElements.get(FORMAT);
    	if (formatVersion instanceof String && ((String) formatVersion).startsWith(V04_START)) {
    		return new ModelDescriptorStardistV04(yamlElements);
    	} else if (formatVersion instanceof String && ((String) formatVersion).startsWith(V05_START)) {
    		return new ModelDescriptorStardistV05(yamlElements);
    	} else if (formatVersion instanceof String)
    		throw new IllegalArgumentException("JDLL only supports the Bioimage.io model specs 0.4 and 0.5.");
    	else {
    		throw new IllegalArgumentException("Incorrect format, missing 'format_version' field.");
    	}
    }

    private static ModelDescriptor fromCellposeMap(Map<String,Object> yamlElements)
    {
    	Object formatVersion = yamlElements.get(FORMAT);
    	if (formatVersion instanceof String && ((String) formatVersion).startsWith(V04_START)) {
    		return new ModelDescriptorCellposeV04(yamlElements);
    	} else if (formatVersion instanceof String && ((String) formatVersion).startsWith(V05_START)) {
    		return new ModelDescriptorCellposeV05(yamlElements);
    	} else if (formatVersion instanceof String)
    		throw new IllegalArgumentException("JDLL only supports the Bioimage.io model specs 0.4 and 0.5.");
    	else {
    		throw new IllegalArgumentException("Incorrect format, missing 'format_version' field.");
    	}
    }

	/**
	 * Get the models at the local repo defined by the argument local repo
	 * @param localRepo
	 * 	String containing the path the directory that contains the model folders
	 * @return a list of the {@link ModelDescriptor}s of the available models
	 */
	public static List<ModelDescriptor> getModelsAtLocalRepo(String localRepo) {
		File repoFile = new File(localRepo);
		if ( !repoFile.isDirectory() )
		{
			return Collections.emptyList();
		}
		return Arrays.asList(repoFile.listFiles()).stream().map(ff -> {
			String rdfPath = ff.getAbsolutePath() + File.separator + Constants.RDF_FNAME;
			if (!(new File(rdfPath).isFile())) return null;
			try {
				return ModelDescriptorFactory.readFromLocalFile(rdfPath);
			} catch (Exception e) {
				return null;
			}
			}).filter(mm -> mm != null).collect(Collectors.toList());
	}

	/**
	 * Get the models at the local repo.
	 * The default local repo is the 'models' folder in the directory where the program is being executed
	 * 
	 * @return a list of the {@link ModelDescriptor}s of the available models
	 */
	public static List<ModelDescriptor> getModelsAtLocalRepo() {
		return getModelsAtLocalRepo(new File("models").getAbsolutePath());
	}
    
    /**
     * Method that checks if a String corresponds to an URL
     * 
     * @param str
     * 	the string that should be possible to convert into URL
     * @return the original string if it does correspond to an URL
     * 	or null if it does not
     */
    protected static String checkUrl(String str) {
		try {
			new URL(str);
			return str;
		} catch (MalformedURLException e) {
			return null;
		}
    }

    /**
     * REturns a {@code List<String>} of data from the yaml file that is supposed
     * to correspond to an URI.
     * @param coverElements
     * 	data from the yaml
     * @return the {@code List<String>}with the URI data
     */
    protected static List<String> buildUrlElements(Object coverElements)
    {
        List<String> covers = new ArrayList<>();
    	if ((coverElements instanceof List<?>)) {
    		List<?> elems = (List<?>) coverElements;
	        for (Object elem : elems)
	        {
	            covers.add((String) elem);
	        }
    	} else if ((coverElements instanceof String)) {
            covers.add((String) coverElements);
    	} else {
    		covers = null;
    	}
    	
        return covers;
    }
    
    /**
     * Check whether the Map corresponds to the yaml file of a Stardist model
     * @param yamlElements
     * 	a Map containing the contents of an rdf.yaml file
     * @return true if the model corresponds to Stardist and false otherwise
     */
    public static boolean isStardist(Map<String,Object> yamlElements) {
    	boolean nameCheck = (yamlElements.get("name") != null) && (yamlElements.get("name") instanceof String) 
    			&& (((String) yamlElements.get("name")).toLowerCase().contains("stardist"));
    	boolean configCheck = (yamlElements.get("config") != null) && (yamlElements.get("config") instanceof Map) 
    			&& (((Map<String, Object>) yamlElements.get("config")).get("stardist") != null);
    	return nameCheck && configCheck;
    }
    
    /**
     * Check whether the Map corresponds to the yaml file of a cellpose model
     * @param yamlElements
     * 	a Map containing the contents of an rdf.yaml file
     * @return true if the model corresponds to cellpose and false otherwise
     */
    public static boolean isCellpose(Map<String,Object> yamlElements) {
    	return false;
    }
}
