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

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.exceptions.ModelSpecsException;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.transformations.PythonTransformation;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.utils.Log;
import io.bioimage.modelrunner.utils.YAMLUtils;


/**
 * A data structure holding a single Bioimage.io pretrained model description. This instances are created by opening a {@code model.yaml} file.
 * More info about the parameters can be found at:
 * https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/model_spec_latest.md
 * 
 * @author Daniel Felipe Gonzalez Obando and Carlos Garcia Lopez de Haro
 */
public class ModelDescriptor
{
    private String format_version;
    private String name;
    private String nickname;
    private String timestamp;
    private String description;
    private String type;
    private List<Author> authors;
    private List<Author> maintainers;
    private List<Author> packaged_by;
    private List<Cite> cite;
    private List<Badge> badges;
    private List<String> tags;
    private String license;
    private String git_repo;
    private String documentation;
    private String rdf_source;
    private List<String> covers;
    private List<SampleImage> sample_inputs;
    private List<SampleImage> sample_outputs;
    private List<TestArtifact> test_inputs;
    private List<TestArtifact> test_outputs;
    private List<TensorSpec> input_tensors;
    private List<TensorSpec> output_tensors;
    private ExecutionConfig config;
    private ModelWeight weights;
    private Map<String, Object> attachments;
    private String download_url;
    private String icon;
    private String version;
    private List<String> links;
    private Map<String, String> parent;
    private boolean isModelLocal;
    private static String fromLocalKey = "fromLocalRepo";
    private static String modelPathKey = "modelPath";
    private static List<String> sampleBioengineModels = Arrays.asList(new String[] {"cell_pose"});//, "inception", "stardist"});
    private String modelID;
    private String localModelPath;
    private boolean supportBioengine = false;

    private ModelDescriptor()
    {
    }
    
    /**
     * Opens the provided file and builds an instance of {@link ModelDescriptor} from it.
     * 
     * @param modelFile
     *        Model descriptor file.
     * @return The instance of the model descriptor.
     * @throws ModelSpecsException if any of the parameters in the rdf.yaml file does not make fit the constraints
     */
    public static ModelDescriptor readFromLocalFile(String modelFile) throws ModelSpecsException
    {
    	return readFromLocalFile(modelFile, true);
    }
    
    /**
     * Opens the provided file and builds an instance of {@link ModelDescriptor} from it.
     * 
     * @param modelFile
     * 	Model descriptor file.
     * @param verbose
     * 	whether to print the path to the file and the time to the console or not
     * @return The instance of the model descriptor.
     * @throws ModelSpecsException if any of the parameters in the rdf.yaml file does not make fit the constraints,
     */
    public static ModelDescriptor readFromLocalFile(String modelFile, boolean verbose) throws ModelSpecsException
    {
    	// Get the date to be able to log with the time
    	if (verbose)
    		System.out.println(Log.gct() + " -- LOCAL: Searching model at " + new File(modelFile).getParent());
    	Map<String, Object> yamlElements;
    	try {
        	yamlElements = YAMLUtils.load(modelFile);
        } catch (IOException ex) {
        	throw new IllegalStateException("", ex);
        }
        yamlElements.put(fromLocalKey, true);
        yamlElements.put(modelPathKey, new File(modelFile).getParent());
        return buildModelDescription(yamlElements);
    }
    
    /**
     * Reads a yaml text String and builds an instance of {@link ModelDescriptor} from it
     * 
     * @param yamlText
     *        text read from a yaml file that contains an rdf.yaml file
     * @return The instance of the model descriptor.
     * @throws ModelSpecsException if any of the parameters in the rdf.yaml file does not make fit the constraints
     */
    public static ModelDescriptor readFromYamlTextString(String yamlText) throws ModelSpecsException
    {
    	return readFromYamlTextString(yamlText, true);
    }
    
    /**
     * Reads a yaml text String and builds an instance of {@link ModelDescriptor} from it.
     * 
     * @param yamlText
     * 	text read from a yaml file that contains an rdf.yaml file
     * @param verbose
     * 	whether to print info about the rdf.yaml that is being read or not
     * @return The instance of the model descriptor.
     * @throws ModelSpecsException if any of the parameters in the rdf.yaml file does not make fit the constraints
     */
    public static ModelDescriptor readFromYamlTextString(String yamlText, boolean verbose) throws ModelSpecsException
    {
    	// Convert the String of text that contains the yaml file into Map
    	Map<String,Object> yamlElements = YAMLUtils.loadFromString(yamlText);
    	// Let the user know the model the plugin is trying to load
    	if (yamlElements.get("name") != null && verbose) {
        	System.out.println(Log.gct() + " -- Bioimage.io: Inspecting model: " + (String) yamlElements.get("name"));
    	} else if (verbose) {
        	System.out.println(Log.gct() + " -- Bioimage.io: Inspecting model defined by: " + yamlText);
    	}
        yamlElements.put(fromLocalKey, false);
        return buildModelDescription(yamlElements);
    }

    @SuppressWarnings("unchecked")
    /**
     * Build a {@link ModelDescriptor} object from a map containing the elements read from
     * a rdf.yaml file
     * @param yamlElements
     * 	map with the information read from a yaml file
     * @return a {@link ModelDescriptor} with the info of a Bioimage.io model
     * @throws ModelSpecsException if any of the parameters in the rdf.yaml file does not make fit the constraints
     */
    private static ModelDescriptor buildModelDescription(Map<String, Object> yamlElements) throws ModelSpecsException
    {
        ModelDescriptor modelDescription = new ModelDescriptor();

        Set<String> yamlFields = yamlElements.keySet();
        String[] yamlFieldsArr = new String[yamlFields.size()];
        Arrays.sort(yamlFields.toArray(yamlFieldsArr));
        for (String field : yamlFieldsArr)
        {
            Object fieldElement = yamlElements.get(field);
            try
            {
                switch (field)
                {
                    case "format_version":
                        modelDescription.format_version = (String) fieldElement;
                        break;
                    case "name":
                        modelDescription.name = (String) fieldElement;
                        break;
                    case "timestamp":
                        modelDescription.timestamp = fieldElement.toString();
                        break;
                    case "description":
                        modelDescription.description = (String) fieldElement;
                        break;
                    case "id":
                        modelDescription.modelID = (String) fieldElement;
                        break;
                    case "authors":
                        modelDescription.authors = buildAuthorElements((List<?>) fieldElement);
                        break;
                    case "maintainers":
                        modelDescription.maintainers = buildAuthorElements((List<?>) fieldElement);
                        break;
                    case "packaged_by":
                        modelDescription.packaged_by = buildAuthorElements((List<?>) fieldElement);
                        break;
                    case "cite":
                        modelDescription.cite = buildCiteElements((List<?>) fieldElement);
                        break;
                    case "parent":
                        modelDescription.parent = (Map<String, String>) fieldElement;
                        break;
                    case "git_repo":
                        modelDescription.git_repo = checkUrl((String) fieldElement);
                        break;
                    case "tags":
                        modelDescription.tags = castListStrings(fieldElement);
                        break;
                    case "links":
                        modelDescription.links = castListStrings(fieldElement);
                        break;
                    case "license":
                        modelDescription.license = (String) fieldElement;
                        break;
                    case "documentation":
                        modelDescription.documentation = (String) fieldElement;
                        break;
                    case "test_inputs":
                        modelDescription.test_inputs = buildTestArtifacts((List<?>) fieldElement);
                        break;
                    case "test_outputs":
                        modelDescription.test_outputs = buildTestArtifacts((List<?>) fieldElement);
                        break;
                    case "sample_inputs":
                        modelDescription.sample_inputs = buildSampleImages((List<?>) fieldElement);
                        break;
                    case "sample_outputs":
                        modelDescription.sample_outputs = buildSampleImages((List<?>) fieldElement);
                        break;
                    case "type":
                        modelDescription.type = (String) fieldElement;
                        break;
                    case "icon":
                        modelDescription.icon = (String) fieldElement;
                        break;
                    case "download_url":
                        modelDescription.download_url = checkUrl((String) fieldElement);
                        break;
                    case "rdf_source":
                        modelDescription.rdf_source = checkUrl((String) fieldElement);
                        break;
                    case "attachments":
                        modelDescription.attachments = (Map<String, Object>) fieldElement;
                        break;
                    case "covers":
                        modelDescription.covers = buildUrlElements((List<?>) fieldElement);
                        break;
                    case "badges":
                        modelDescription.badges = buildBadgeElements((List<?>) fieldElement);
                        break;
                    case "inputs":
                    	modelDescription.setInputTensors(buildInputTensors((List<?>) yamlElements.get(field)));
                        break;
                    case "outputs":
                        modelDescription.setOutputTensors(buildOutputTensors((List<?>) yamlElements.get(field)));
                        break;
                    case "config":
                        modelDescription.config = buildConfig((Map<String, Object>) yamlElements.get(field));
                        break;
                    case "weights":
                        modelDescription.weights = buildWeights((Map<String, Object>) yamlElements.get(field));
                        break;
                    case "fromLocalRepo":
                        modelDescription.isModelLocal = (boolean) fieldElement;
                        break;
                    case "modelPath":
                        modelDescription.localModelPath = (String) fieldElement;
                        break;
                    default:
                        break;
                }
            }
            catch (IOException e)
            {
                throw new ModelSpecsException("Invalid model element: " + field + "->" + e.getMessage());
            }
        }
        Object bio = modelDescription.config.getSpecMap().get("bioimageio");
        if ((bio != null) && (bio instanceof Map))
        	modelDescription.nickname = (String) (((Map<String, Object>) bio).get("nickname"));
        modelDescription.addBioEngine();
        if (modelDescription.localModelPath == null)
        	return modelDescription;
    	modelDescription.addModelPath(new File(modelDescription.localModelPath).toPath());
    	SpecialModels.checkSpecialModels(modelDescription);
    	return modelDescription;
    }
    
    /**
     * Every model in the bioimage.io can be run in the BioEngine as long as it is in the
     * collections repo: 
     * https://github.com/bioimage-io/collection-bioimage-io/blob/e77fec7fa4d92d90c25e11331a7d19f14b9dc2cf/rdfs/10.5281/zenodo.6200999/6224243/rdf.yaml
     * @throws ModelSpecsException servers do not correspond to an actual url
     */
    private void addBioEngine() throws ModelSpecsException {
		// TODO decide what to do with servers. Probably need permissions / Implement authentication
    	if (getName().equals("cellpose-python")) {
    		supportBioengine = true;
			return;
	    } else if (getName().equals("bestfitting-inceptionv3-single-cell")) {
			return;
	    } else if (getName().equals("stardist")) {
    		supportBioengine = true;
			return;
	    } else if (modelID == null) {
    		supportBioengine = false;
	    	return;
	    }
    	try {
			supportBioengine =  BioimageioRepo.isModelOnTheBioengineById(modelID);
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
    
    /**
     * TODO
     * TODO
     * TODO ADD BIOENGINE SOON
     * Method that retrieves the sample BioEngine models that Icy provides as an example
     * to test the BioEngine
     * @return a list with sample biengine models
     */
    public static ArrayList<Entry<Path, ModelDescriptor>> addSampleBioEngineModels() {
    	ArrayList<Entry<Path, ModelDescriptor>> sampleModels = new ArrayList<Entry<Path, ModelDescriptor>>();
    	for (String sample : sampleBioengineModels) {
			 try {
	        	InputStream inputStream = ModelDescriptor.class.getClassLoader()
	        												.getResourceAsStream(sample + ".yaml");
	        	ByteArrayOutputStream result = new ByteArrayOutputStream();
				 byte[] buffer = new byte[1024];
				 for (int length; (length = inputStream.read(buffer)) != -1; ) {
				     result.write(buffer, 0, length);
				 }
				 // StandardCharsets.UTF_8.name() > JDK 7
				 String txt = result.toString("UTF-8");
				 result.close();
				 inputStream.close();
				 // TODO sampleModels.add(CollectionUtils.createEntry(Paths.get(sample), loadFromYamlTextString(txt)));
			} catch (Exception e) {
				e.printStackTrace();
				System.out.println(Log.gct() + " -- BioEngine: unable to load sample model " + sample);
			}
    	}
    	return sampleModels;
    }
    
    /**
     * MAke sure that an object that is supposed to be a List<String>
     * is actually a List<String>
     * @param list
     * 	the possible List<String>
     * @return a List<String> or null if the Object was not an instance of a List<String>
     */
    private static List<String> castListStrings(Object list) {
    	List<String> out = null;
    	if (list instanceof List<?>) {
    		out = new ArrayList<String>();
    		out = (List<String>) list;
    	} else if (list instanceof String) {
    		out = new ArrayList<String>();
    		out.add((String) list);
    	}
    	return out;
    }
    
    /**
     * Create a list with the authors of teh model as read from the rdf.yaml file
     * @param authElements
     * 	a raw list with the info about the authors
     * @return a list with the info about the authors packaged in the {@link Author} object
     */
    private static List<Author> buildAuthorElements(List<?> authElements)
    {
        List<Author> authors = new ArrayList<>();
        for (Object elem : authElements)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            @SuppressWarnings("unchecked")
            Map<String, String> dict = (Map<String, String>) elem;
            authors.add(Author.build(dict.get("affiliation"), dict.get("email"), dict.get("github_user"), dict.get("name"), dict.get("orcid")));
        }
        return authors;
    }
    
    /**
     * Create a list with the citations of the model as read from the rdf.yaml file
     * @param citeElements
     * 	a raw list with the info about the citations
     * @return a list with the info about the citations packaged in the {@link Cite} object
     */
    private static List<Cite> buildCiteElements(List<?> citeElements) throws MalformedURLException
    {
    	if (!(citeElements instanceof List<?>))
    		return null;
        List<Cite> cites = new ArrayList<>();
        for (Object elem : citeElements)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            @SuppressWarnings("unchecked")
            Map<String, Object> dict = (Map<String, Object>) elem;
            cites.add(Cite.build((String) dict.get("text"), (String) dict.get("doi"), (String) dict.get("url")));
        }
        return cites;
    }

    /**
     * REturns a List<String> of data from the yaml file that is supposed
     * to correspond to an URI.
     * @param coverElements
     * 	data from the yaml
     * @return the List<String> with the URI data
     */
    private static List<String> buildUrlElements(Object coverElements)
    {
        List<String> covers = new ArrayList<>();
    	if ((coverElements instanceof List<?>)) {
    		List<?> elems = (List<?>) coverElements;
	        for (Object elem : elems)
	        {
	        	if (checkUrl((String) elem) == null)
	        		continue;
	            covers.add((String) elem);
	        }
    	} else if ((coverElements instanceof String) && checkUrl((String) coverElements) != null) {
            covers.add((String) coverElements);
    	} else {
    		covers = null;
    	}
    	
        return covers;
    }

    /**
     * REturns a List<SampleInputs> of the sample images that are packed in the model
     * folder as tifs and that are specified in the rdf.yaml file
     * @param coverElements
     * 	data from the yaml
     * @return the List<SampleInputs> with the sample images data
     */
    private static List<SampleImage> buildSampleImages(Object coverElements)
    {
        List<SampleImage> covers = new ArrayList<>();
    	if ((coverElements instanceof List<?>)) {
    		List<?> elems = (List<?>) coverElements;
	        for (Object elem : elems)
	        {
	        	if (!(elem instanceof String))
	        		continue;
	        	covers.add(SampleImage.build((String) elem));
	        }
    	} else if ((coverElements instanceof String)) {
            covers.add(SampleImage.build((String) coverElements));
    	}   	
        return covers.stream().filter(i -> i != null).collect(Collectors.toList());
    }

    /**
     * REturns a List<TestArtifact> of the npy artifacts that are packed in the model
     * folder as input and output test objects
     * @param coverElements
     * 	data from the yaml
     * @return the List<TestArtifact> with the sample images data
     */
    private static List<TestArtifact> buildTestArtifacts(Object coverElements)
    {
        List<TestArtifact> covers = new ArrayList<>();
    	if ((coverElements instanceof List<?>)) {
    		List<?> elems = (List<?>) coverElements;
	        for (Object elem : elems)
	        {
	        	if (!(elem instanceof String))
	        		continue;
	        	covers.add(TestArtifact.build((String) elem));
	        }
    	} else if ((coverElements instanceof String)) {
            covers.add(TestArtifact.build((String) coverElements));
    	}   	
        return covers.stream().filter(i -> i != null).collect(Collectors.toList());
    }

    private static List<Badge> buildBadgeElements(List<?> coverElements)
    {
    	if (!(coverElements instanceof List<?>))
    		return null;
        List<Badge> badges = new ArrayList<>();
        for (Object elem : coverElements)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            Map<String, Object> dict = (Map<String, Object>) elem;
        	badges.add(Badge.build((String) dict.get("label"), (String) dict.get("icon"), (String) dict.get("url")));
        }
        return badges;
    }

    @SuppressWarnings("unchecked")
    private static List<TensorSpec> buildInputTensors(List<?> list) throws ModelSpecsException
    {
    	if (!(list instanceof List<?>))
    		return null;
        List<TensorSpec> tensors = new ArrayList<>(list.size());
        for (Object elem : list)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            tensors.add(TensorSpec.build((Map<String, Object>) elem, true));
        }
        return tensors;
    }
    
    /**
     * Sets the input tensors of the model specs. Also adds the tiling
     * @param inputTensors
     * 	the input tensors
     */
    public void setInputTensors(List<TensorSpec> inputTensors) {
    	boolean tiling = getConfig() == null ? true : 
    				(getConfig().getDeepImageJ() == null ? true : getConfig().getDeepImageJ().isAllowTiling());
    	for (TensorSpec tt : inputTensors)
    		tt.setTiling(tiling);
    	this.input_tensors = inputTensors;
    }

    @SuppressWarnings("unchecked")
    private static List<TensorSpec> buildOutputTensors(List<?> list) throws ModelSpecsException
    {
    	if (!(list instanceof List<?>))
    		return null;
        List<TensorSpec> tensors = new ArrayList<>(list.size());
        for (Object elem : list)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            tensors.add(TensorSpec.build((Map<String, Object>) elem, false));
        }
        return tensors;
    }
    
    /**
     * This method sets the output tensors and creates a total halo that is used
     * by the inputs
     * @param outputTensors
     */
    private void setOutputTensors(List<TensorSpec> outputTensors) {
		this.output_tensors = outputTensors;
		float[] halo = calculateTotalInputHalo();
		for (TensorSpec inp : this.input_tensors)
			inp.setTotalHalo(halo);
    }
    
    /**
     * Calculate the total input halo once the output tensors are set.
     * NOte that the total halo is calculated following "xyczb" axes order,
     * not the input axes order, as for several inputs the axes order might change
     * for each of them
     * @return the total input halo in "xyczb" axes order
     */
    private float[] calculateTotalInputHalo() {
    	String[] targetForm = "XYCZB".split("");
		float[] halo = new float[targetForm.length];
		for (TensorSpec out: output_tensors) {
			for (int i = 0; i < targetForm.length; i ++) {
				int ind = out.getAxesOrder().toUpperCase().indexOf(targetForm[i]);
				if (ind == -1)
					continue;
				float inputHalo = out.getHalo()[ind];
				// No halo in channel C because of offset
				float inputOffset = -1 * out.getShape().getOffset()[ind];
				if (targetForm[i].toLowerCase().equals("c"))
					inputOffset = 0;
				float possibleHalo = (inputHalo + inputOffset) / out.getShape().getScale()[ind];
				if (possibleHalo > halo[i])
					halo[i] = possibleHalo;
			}
		}
		return halo;
    }

    private static ExecutionConfig buildConfig(Map<String, Object> yamlFieldElements)
    {
        return ExecutionConfig.build(yamlFieldElements);
    }

    private static ModelWeight buildWeights(Map<String, Object> yamlFieldElements)
    {
        return ModelWeight.build(yamlFieldElements);
    }
    
    /**
     * Method that checks if a String corresponds to an URL
     * 
     * @param str
     * 	the string that should be possible to convert into URL
     * @return the original string if it does correspond to an URL
     * 	or null if it does not
     */
    public static String checkUrl(String str) {
		try {
			URL url = new URL(str);
			return str;
		} catch (MalformedURLException e) {
			return null;
		}
    }
    
    /**
     * Create a set of specifications about the basic info of the model: name od the model, authors,
     * references and Deep Learning framework
     * @return a set of specs for the model
     */
    public String buildBasicInfo() {
    	String info = "";
    	// Display the name
    	info += "&nbsp -Name: " + this.getName().toUpperCase() + "<br>";
    	// Create authors string
    	String auth = "[";
    	for (Author author : this.authors)
    		auth += (author.getName() != null ? author.getName() : "null") + "; ";
    	// Remove the "; " characters at the end and add "]"
    	if (auth.length() < 3)
    		auth = "[]";
    	else
    		auth = auth.substring(0, auth.length() - 2) + "]";
    	// Display the authors
    	info += "&nbsp -Authors: " + auth + "<br>";
    	// Create the references String
    	String refs = "[";
    	if (cite != null) {
	    	for (Cite citation : this.cite)
	    		refs += (citation.getText() != null ? citation.getText() : (citation.getDoi() != null ? citation.getDoi() : "null")) + "; ";
    	}
    	// Remove the "; " characters at the end and add "]"
    	refs = refs.length() > 2 ? refs.substring(0, refs.length() - 2) + "]" : "[]";
    	// Display the references
    	info += "&nbsp -References: " + refs + "<br>";
    	// Add the model description
    	if (this.getDescription() != null)
    		info += "&nbsp -Description: " + this.getDescription() + "<br>";
    	info += "<br>";
    	// Add the location of the model in the local machine if it exists
    	String location = localModelPath != null ? localModelPath : rdf_source;
    	if (location == null)
    		info += "&nbsp -Location: " + "UNKNOWN" + "<br>";
    	else
    		info += "&nbsp -Location: " + location + "<br>";
    	// Display the frameworks available for this model
    	info += "&nbsp -Engine: " + this.weights.getSupportedWeightNamesAndVersion().toString() + "<br>";
    	// Display the model id
    	info += "&nbsp -ID: " + this.modelID + "<br>";
    	info += "<br>";
    	return info;
    }
    
    /**
     * Write the tiling specs for the model
     * @return the tiling specs for the model
     */
    public String buildTilingInfo() {
    	String info = "&nbsp ----TILING SPECIFICATIONS----"  + "<br>";
    	HashMap<String, String> dimMeaning = new HashMap<String, String>(){{
    		put("H", "height"); put("X", "width");
    		put("Z", "depth"); put("C", "channels");
    		}};
    	// Create the String that explains the dimensions letters
    	info += "&nbsp Y: height, X: width, Z: depth, C: channels" + "<br>";
    	// Add tiling info foe each of the arguments
    	info += "&nbsp -input images:" + "<br>";
    	for (TensorSpec inp : this.input_tensors) {
    		info += "&ensp -" + inp.getName() + ":<br>";
    		String[] dims = inp.getAxesOrder().toUpperCase().split("");
    		String minString = "&emsp -minimum size: ";
    		String stepString = "&emsp -step: ";
    		for (int i = 0; i < dims.length; i ++) {
    			minString += dims[i] + ": " + inp.getShape().getTileMinimumSize()[i] + ", ";
    			stepString += dims[i] + ": " + inp.getShape().getTileStep()[i] + ", ";
    		}
    		// Remove the "; " characters at the end and add "]"
    		minString = minString.substring(0, minString.length() - 2) + "<br>";
    		stepString = stepString.substring(0, stepString.length() - 2) + "<br>";
    		info += minString;
    		info += stepString;
    	}
    	// Add small explanation
    	info += "&nbsp Each dimension is calculated as:" + "<br>";
    	info += "&ensp " + "tile_size = minimum_size + n * step, where n >= 0" + "<br>";
    	return info;
    }
    
    /**
     * Create specifications of the model containing the most important
     * info that is going to be displayed on the DeepIcy plugin
     * @return a String with the most important info
     */
    public String buildInfo() {
    	String basicInfo = buildBasicInfo();
    	String tilingInfo = buildTilingInfo();    		
    	return basicInfo + tilingInfo;
    }

    /**
     * @return The version of the format used in the descriptor file.
     */
    public String getFormatVersion()
    {
        return format_version;
    }

    /**
     * @return The name of this model.
     */
    public String getName()
    {
        return name;
    }

    /**
     * @return The nickname of this model.
     */
    public String getNickname()
    {
        return nickname;
    }

    /**
     * @return The ID of this model.
     */
    public String getModelID()
    {
        return modelID;
    }

    /**
     * @return The creation timestamp of this model.
     */
    public String getTimestamp()
    {
        return timestamp;
    }

    /**
     * @return The description of this model.
     */
    public String getDescription()
    {
        return description;
    }

    /**
     * @return The list of authors for this model.
     */
    public List<Author> getAuthors()
    {
        return authors;
    }

    /**
     * @return The list of citations for this model.
     */
    public List<Cite> getCite()
    {
        return cite;
    }

    /**
     * @return The URL of the git repository of this model.
     */
    public String getGitRepo()
    {
        return git_repo;
    }

    /**
     * @return The list of tags associated with this model.
     */
    public List<String> getTags()
    {
        return tags;
    }

    /**
     * @return The license description for this model.
     */
    public String getLicense()
    {
        return license;
    }

    /**
     * @return the type of Bioimage.io artifact that the rdf.yaml
     * refers to. It can be model, dataset, application...
     */
    public String getType()
    {
        return type;
    }

    /**
     * @return The documentation text associated to this model.
     */
    public String getDocumentation()
    {
        return documentation;
    }

    /**
     * @return The list of URIs of the covers for this model.
     */
    public List<String> getCovers()
    {
        return covers;
    }

    /**
     * @return The list of input tensor specification instances for this model.
     */
    public List<TensorSpec> getInputTensors()
    {
        return input_tensors;
    }

    /**
     * Searches for an input tensor with the given name.
     * 
     * @param name
     *        Name of the tensor.
     * @return The tensor with the provided name. null is returned if no tensor is found or if the input tensors list is not initialized.
     */
    public TensorSpec findInputTensor(String name)
    {
        if (input_tensors == null)
        {
            return null;
        }

        return input_tensors.stream()
                .filter(t -> t.getName().equals(name))
                .findAny().orElse(null);
    }

    /**
     * Searches for an output tensor with the given name.
     * 
     * @param name
     *        Name of the tensor.
     * @return The tensor with the provided name. null is returned if no tensor is found or if the output tensors list is not initialized.
     */
    public TensorSpec findOutputTensor(String name)
    {
        if (output_tensors == null)
        {
            return null;
        }

        return output_tensors.stream()
                .filter(t -> t.getName().equals(name))
                .findAny().orElse(null);
    }

    /**
     * Searches for an input tensor with the given name in the given list.
     * 
     * @param name
     *        Name of the tensor.
     * @param tts
     * 		  list of tensors where to look for the wanted name
     * @return The tensor with the provided name. null is returned if no tensor is found or if the input tensors list is not initialized.
     */
    public static TensorSpec findTensorInList(String name, List<TensorSpec> tts)
    {
        if (tts == null)
        {
            return null;
        }

        return tts.stream()
                .filter(t -> t.getName().equals(name))
                .findAny().orElse(null);
    }

    /**
     * @return The list of output tensor specification instances for this model.
     */
    public List<TensorSpec> getOutputTensors()
    {
        return output_tensors;
    }

    /**
     * @return The execution configuration instance for this model.
     */
    public ExecutionConfig getConfig()
    {
        return config;
    }

    /**
     * @return The model weights instance for this model.
     */
    public ModelWeight getWeights()
    {
        return weights;
    }

    /**
	 * @return the sample_inputs
	 */
	public List<SampleImage> getSampleInputs() {
		if (sample_inputs == null) 
			sample_inputs = new ArrayList<SampleImage>();
		return sample_inputs;
	}

	@Override
    public String toString()
    {
        return "ModelDescription {formatVersion=" + format_version + ", name=" + name + ", timestamp=" + timestamp
                + ", description=" + description + ", authors=" + authors + ", cite=" + cite + ", gitRepo=" + git_repo
                + ", tags=" + tags + ", license=" + license + ", documentation=" + documentation + ", covers=" + covers
                + ", inputTensors=" + input_tensors + ", outputTensors=" + output_tensors + ", config=" + config
                + ", weights=" + weights + "}";
    }

	/**
	 * @return the maintainers
	 */
	public List<Author> getMaintainers() {
		return maintainers;
	}

	/**
	 * @return the packaged_by
	 */
	public List<Author> getPackagedBy() {
		return packaged_by;
	}

	/**
	 * @return the badges
	 */
	public List<Badge> getBadges() {
		if (badges == null) 
			badges = new ArrayList<Badge>();
		return badges;
	}

	/**
	 * @return the sample_outputs
	 */
	public List<SampleImage> getSampleOutputs() {
		if (sample_outputs == null) 
			sample_outputs = new ArrayList<SampleImage>();
		return sample_outputs;
	}

	/**
	 * @return the test_inputs
	 */
	public List<TestArtifact> getTestInputs() {
		if (test_inputs == null) 
			test_inputs = new ArrayList<TestArtifact>();
		return test_inputs;
	}

	/**
	 * @return the test_outputs
	 */
	public List<TestArtifact> getTestOutputs() {
		if (test_outputs == null) 
			test_outputs = new ArrayList<TestArtifact>();
		return test_outputs;
	}

	/**
	 * @return the attachments
	 */
	public Map<String, Object> getAttachments() {
		return attachments;
	}

	/**
	 * @return the download_url
	 */
	public String getDownloadUrl() {
		return download_url;
	}

	/**
	 * @return the rdf_source
	 */
	public String getRDFSource() {
		return rdf_source;
	}

	/**
	 * @return the icon
	 */
	public String getIcon() {
		return icon;
	}

	/**
	 * @return the version
	 */
	public String getVersion() {
		return version;
	}

	/**
	 * @return the links
	 */
	public List<String> getLinks() {
		return links;
	}

	/**
	 * @return the parent
	 */
	public Map<String, String> getParent() {
		return parent;
	}
	
	/**
	 * Mark the model as downloaded or not. This method is useful for when the
	 * user selects a model from the BioImage.io
	 * @param dd
	 * 	whether the model is already downloaded or not
	 */
	public void setDownloaded(boolean dd) {
		isModelLocal = dd;
	}
	
	/**
	 * Whether the model is already in the local repo or it has to be downloaded
	 * @return true if the model is already installed or false otherwise
	 */
	public boolean isModelInLocalRepo() {
		return isModelLocal;
	}
	
	/**
	 * Add the path where the local model is stored to the model descriptor
	 * @param modelBasePath
	 * 	the path to the model in the local machine
	 */
	public void addModelPath(Path modelBasePath) {
		if (!modelBasePath.toFile().exists()) {
			throw new IllegalArgumentException("The path '" 
					 + modelBasePath.toString() + "' does not exist in the computer.");
		}
		localModelPath = modelBasePath.toString();
		if (sample_inputs != null)
			sample_inputs.stream().forEach(i -> i.addLocalModelPath(modelBasePath));
		if (sample_outputs != null)
			sample_outputs.stream().forEach(i -> i.addLocalModelPath(modelBasePath));
		if (test_inputs != null)
			test_inputs.stream().forEach(i -> i.addLocalModelPath(modelBasePath));
		if (test_outputs != null)
			test_outputs.stream().forEach(i -> i.addLocalModelPath(modelBasePath));
	}
	
	/**
	 * Return String to path where the model is stored
	 * @return String directory where the model is stored
	 */
	public String getModelPath() {
		return this.localModelPath;
	}
	
	/**
	 * Method that returns whether tiling is allowed or not for the model
	 * @return true if tiling is allowed and false otherwise
	 */
	public boolean isTilingAllowed() {
		if (this.config == null)
			return true;
		else if (this.config.getDeepImageJ() == null)
			return true;
		else 
			return this.getConfig().getDeepImageJ().isAllowTiling();
	}
	
	/**
	 * Method that returns whether the model is pyramidal or not
	 * @return true if the model is pyramidal, false otherwise
	 */
	public boolean isPyramidal() {
		if (this.config == null)
			return false;
		else if (this.config.getDeepImageJ() == null)
			return false;
		else 
			return this.getConfig().getDeepImageJ().isPyramidalModel();
	}
	
	/**
	 * 
	 * @return whether the model can be run on the bioengino or not
	 */
	public boolean canRunOnBioengine() {
		return this.supportBioengine;
	}

	/**
	 * Get the models at the local repo defined by the argument local repo
	 * @param localRepo
	 * 	String containing the path the directory that contains the model folders
	 * @return a list of the {@link ModelDescriptor}s of the available models
	 */
	public static List<ModelDescriptor> getModelsAtLocalRepo(String localRepo) {
		File repoFile = new File(localRepo);
		if (!repoFile.isDirectory())
			throw new IllegalArgumentException("The provided path is not a valid directory: " + localRepo);
		return Arrays.asList(repoFile.listFiles()).stream().map(ff -> {
			try {
				return ModelDescriptor.readFromLocalFile(ff.getAbsolutePath() + File.separator + Constants.RDF_FNAME, false);
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
}
