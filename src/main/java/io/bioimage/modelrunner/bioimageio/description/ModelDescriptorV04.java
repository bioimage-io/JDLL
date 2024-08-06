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
import java.net.MalformedURLException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.exceptions.ModelSpecsException;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.utils.Constants;


/**
 * A data structure holding a single Bioimage.io pretrained model description. This instances are created by opening a {@code model.yaml} file.
 * More info about the parameters can be found at:
 * https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/model_spec_latest.md
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public class ModelDescriptorV04 implements ModelDescriptor
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
    private List<String> covers;
    private List<TensorSpec> input_tensors;
    private List<TensorSpec> output_tensors;
    private ExecutionConfig config;
    private ModelWeight weights;
    private Map<String, Object> attachments;
    private String download_url;
    private String version;
    private List<String> links;
    private Map<String, String> parent;
    private boolean isModelLocal;
    private static String fromLocalKey = "fromLocalRepo";
    private static String modelPathKey = "modelPath";
    private String modelID;
    private String localModelPath;
    private boolean supportBioengine = false;
    
    private static BioimageioRepo BMZ_REPO;

    private ModelDescriptorV04()
    {
    }

    @SuppressWarnings("unchecked")
    /**
     * Build a {@link ModelDescriptorV04} object from a map containing the elements read from
     * a rdf.yaml file
     * @param yamlElements
     * 	map with the information read from a yaml file
     * @return a {@link ModelDescriptorV04} with the info of a Bioimage.io model
     * @throws ModelSpecsException if any of the parameters in the rdf.yaml file does not make fit the constraints
     */
    protected static ModelDescriptorV04 buildModelDescription(Map<String, Object> yamlElements) throws ModelSpecsException
    {
        ModelDescriptorV04 modelDescription = new ModelDescriptorV04();

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
	                case "version":
	                    modelDescription.version = (String) fieldElement;
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
                    	modelDescription.modelID = findID(yamlElements);
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
                        modelDescription.git_repo = ModelDescriptorFactory.checkUrl((String) fieldElement);
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
                    case "type":
                        modelDescription.type = (String) fieldElement;
                        break;
                    case "attachments":
                        modelDescription.attachments = (Map<String, Object>) fieldElement;
                        break;
                    case "covers":
                        modelDescription.covers = ModelDescriptorFactory.buildUrlElements((List<?>) fieldElement);
                        break;
                    case "badges":
                        modelDescription.badges = buildBadgeElements((List<?>) fieldElement);
                        break;
                    case "inputs":
                    	modelDescription.input_tensors = buildInputTensors((List<?>) yamlElements.get(field));
                        break;
                    case "outputs":
                        modelDescription.output_tensors = buildOutputTensors((List<?>) yamlElements.get(field));
                        modelDescription.calculateTotalInputHalo();
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
        
        modelDescription.addSampleAndTestImages(yamlElements);
        
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
    
    private void addSampleAndTestImages(Map<String, Object> yamlElements) {
        List<SampleImage> sampleInputs = buildSampleImages((List<?>) yamlElements.get("sample_inputs"));
        List<SampleImage> sampleOutputs = buildSampleImages((List<?>) yamlElements.get("sample_outputs"));

        List<TestArtifact> testInputs = buildTestArtifacts((List<?>) yamlElements.get("test_inputs"));
        List<TestArtifact> testOutputs = buildTestArtifacts((List<?>) yamlElements.get("test_outputs"));

        for (int i = 0; i < sampleInputs.size(); i ++) {
        	TensorSpecV04 tt = (TensorSpecV04) this.input_tensors.get(i);
        	tt.sampleTensorName = sampleInputs.get(i).getName();
        }
        for (int i = 0; i < testInputs.size(); i ++) {
        	TensorSpecV04 tt = (TensorSpecV04) this.input_tensors.get(i);
        	tt.testTensorName = testInputs.get(i).getName();
        }
        
        for (int i = 0; i < sampleOutputs.size(); i ++) {
        	TensorSpecV04 tt = (TensorSpecV04) this.output_tensors.get(i);
        	tt.sampleTensorName = sampleOutputs.get(i).getName();
        }
        for (int i = 0; i < testOutputs.size(); i ++) {
        	TensorSpecV04 tt = (TensorSpecV04) this.output_tensors.get(i);
        	tt.testTensorName = testOutputs.get(i).getName();
        }
        
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
    
    private static String findID(Map<String, Object> yamlElements) {

    	if (yamlElements.get("config") instanceof Map) {
    		Map<String, Object> configMap = (Map<String, Object>) yamlElements.get("config");
    		if (configMap.get("bioimageio") instanceof Map) {
    			Map<String, Object> bioimageMap = (Map<String, Object>) configMap.get("bioimageio");
    			if (bioimageMap.get("nickname") != null)
    				return (String) bioimageMap.get("nickname");
    		}
    	}
    	return (String) yamlElements.get("id");
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
            tensors.add(new TensorSpecV04((Map<String, Object>) elem, true));
        }
        return tensors;
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
            tensors.add(new TensorSpecV04((Map<String, Object>) elem, false));
        }
        return tensors;
    }
    
    /**
     * Calculate the total input halo once the output tensors are set.
     * NOte that the total halo is calculated following "xyczb" axes order,
     * not the input axes order, as for several inputs the axes order might change
     * for each of them
     * @return the total input halo in "xyczb" axes order
     */
    private void calculateTotalInputHalo() {
		for (TensorSpec out: output_tensors) {
			for (Axis ax : out.getAxesInfo().getAxesList()) {
				int axHalo = ax.getHalo();
				if (axHalo == 0)
					continue;
				String ref = ax.getReferenceTensor();
				if (ref == null) {
					this.input_tensors.stream().forEach( tt -> {
						AxisV04 inAx = (AxisV04) tt.getAxesInfo().getAxesList().stream()
						.filter(xx -> xx.getAxis().equals(ax.getAxis()))
						.findFirst().orElse(null);
						if (inAx == null || inAx.getHalo() > axHalo) return;
						inAx.halo = axHalo;
					});
					return;
				}
				
				double axScale = ax.getScale();
				double axOffset = ax.getOffset();
				double nHalo = (axHalo + axOffset) / axScale;
				AxisV04 inAx = (AxisV04) this.findInputTensor(ref).getAxesInfo().getAxis(ax.getReferenceAxis());

				if (inAx == null || inAx.getHalo() > nHalo) return;
				inAx.halo = (int) nHalo;
			}
		}
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
                .filter(t -> t.getTensorID().equals(name))
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
                .filter(t -> t.getTensorID().equals(name))
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
	 * @return the attachments
	 */
	public Map<String, Object> getAttachments() {
		return attachments;
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

	@Override
	public String buildBasicInfo() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String buildTilingInfo() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String buildInfo() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void addModelPath(Path modelBasePath) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getModelURL() {
		if (this.download_url == null && BMZ_REPO == null) {
			BMZ_REPO = BioimageioRepo.connect();
		}
		
		if (this.download_url == null)
			this.download_url = BMZ_REPO.getModelRdfUrl(modelID, version);
		return this.download_url;
	}

	@Override
	public String getRDFSource() {
		return getModelURL() + Constants.RDF_FNAME;
	}
}
