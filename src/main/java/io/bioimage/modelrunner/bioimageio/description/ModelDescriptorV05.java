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
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
public class ModelDescriptorV05 implements ModelDescriptor
{
    private String format_version;
    private String name;
    private String download_url;
    private String timestamp;
    private String description;
    private String type;
    private String git_repo;
    private List<Author> authors;
    private List<Author> maintainers;
    private List<Author> packaged_by;
    private List<Cite> cite;
    private List<String> tags;
    private String license;
    private String documentation;
    private List<String> covers;
    private List<TensorSpec> input_tensors;
    private List<TensorSpec> output_tensors;
    private ExecutionConfig config;
    private ModelWeight weights;
    private Map<String, Object> attachments;
    private String version;
    private List<String> links;
    private static String fromLocalKey = "fromLocalRepo";
    private static String modelPathKey = "modelPath";
    private String modelID;
    private String localModelPath;
    private boolean supportBioengine = false;
	private  Map<String, Object> yamlElements;

	protected ModelDescriptorV05(Map<String, Object> yamlElements) throws ModelSpecsException
    {
    	this.yamlElements = yamlElements;
    	buildModelDescription();
    }

    @SuppressWarnings("unchecked")
    /**
     * Build a {@link ModelDescriptorV05} object from a map containing the elements read from
     * a rdf.yaml file
     * @param yamlElements
     * 	map with the information read from a yaml file
     * @return a {@link ModelDescriptorV05} with the info of a Bioimage.io model
     * @throws ModelSpecsException if any of the parameters in the rdf.yaml file does not make fit the constraints
     */
    protected void buildModelDescription() throws ModelSpecsException
    {

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
                        format_version = (String) fieldElement;
                        break;
                    case "name":
                        name = (String) fieldElement;
                        break;
                    case "timestamp":
                        timestamp = fieldElement.toString();
                        break;
                    case "description":
                        description = (String) fieldElement;
                        break;
                    case "id":
                        modelID = (String) fieldElement;
                        break;
                    case "authors":
                        buildAuthors();
                        break;
                    case "maintainers":
                        buildAuthors();
                        break;
                    case "packaged_by":
                        buildAuthors();
                        break;
                    case "cite":
                        buildCiteElements();
                        break;
                    case "tags":
                        tags = castListStrings(fieldElement);
                        break;
                    case "links":
                        links = castListStrings(fieldElement);
                        break;
                    case "license":
                        license = (String) fieldElement;
                        break;
                    case "documentation":
                        documentation = (String) fieldElement;
                        break;
                    case "git_repo":
                    	git_repo = (String) fieldElement;
                        break;
                    case "type":
                        type = (String) fieldElement;
                        break;
                    case "attachments":
                        // TODO createAttachments();
                        break;
                    case "covers":
                    	covers = castListStrings(yamlElements.get(field));
                        break;
                    case "inputs":
                    	input_tensors = buildInputTensors((List<?>) yamlElements.get(field));
                        break;
                    case "outputs":
                        output_tensors = buildOutputTensors((List<?>) yamlElements.get(field));
                        calculateTotalInputHalo();
                        break;
                    case "config":
                        config = buildConfig((Map<String, Object>) yamlElements.get(field));
                        break;
                    case "weights":
                        weights = buildWeights((Map<String, Object>) yamlElements.get(field));
                        break;
                    case "modelPath":
                        localModelPath = (String) fieldElement;
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
        if (modelID == null) {
        	modelID = findID(yamlElements);
        }
        if (localModelPath != null && modelID == null)
        	return;
        if (modelID.length() - modelID.replace("/", "").length() >= 2 
				&& modelID.substring(modelID.indexOf("/") + 1).indexOf("/") - modelID.indexOf("/") > 2 )
        	modelID = modelID.substring(0, modelID.indexOf("/") + modelID.substring(modelID.indexOf("/") + 1).indexOf("/") + 1);
        addBioEngine();
        if (localModelPath == null)
        	return;
    	// TODO SpecialModels.checkSpecialModels(null);
    }
    
    @SuppressWarnings("unchecked")
	private static String findID(Map<String, Object> yamlElements) {

    	if (yamlElements.get("config") != null && yamlElements.get("config") instanceof Map) {
    		Map<String, Object> configMap = (Map<String, Object>) yamlElements.get("config");
    		if (configMap.get("bioimageio") != null && configMap.get("bioimageio") instanceof Map) {
    			Map<String, Object> bioimageMap = (Map<String, Object>) configMap.get("bioimageio");
    			if (bioimageMap.get("nickname") != null)
    				return (String) bioimageMap.get("nickname");
    		}
    	}
    	return (String) yamlElements.get("id");
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
     * @return a list with the info about the authors packaged in the {@link Author} object
     */
    private void buildAuthors()
    {
        List<Author> authors = new ArrayList<Author>();
    	Object authorsElems = this.yamlElements.get("authors");
    	if (authorsElems == null || !(authorsElems instanceof List)) {
            this.authors = authors;
            return;
    	}
        for (Object elem : (List<Object>) authorsElems)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            @SuppressWarnings("unchecked")
            Map<String, String> dict = (Map<String, String>) elem;
            authors.add(Author.build(dict.get("affiliation"), dict.get("email"), dict.get("github_user"), dict.get("name"), dict.get("orcid")));
        }
        this.authors = authors;
    }
    
    /**
     * Create a list with the citations of the model as read from the rdf.yaml file
     * @return a list with the info about the citations packaged in the {@link Cite} object
     */
    private void buildCiteElements() throws MalformedURLException
    {
    	Object citeElements = this.yamlElements.get("cite");
        List<Cite> cites = new ArrayList<Cite>();
    	if (citeElements == null || !(citeElements instanceof List<?>)) {
    		this.cite = cites;
    		return;
    	}
        for (Object elem : (List) citeElements)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            @SuppressWarnings("unchecked")
            Map<String, Object> dict = (Map<String, Object>) elem;
            cites.add(Cite.build((String) dict.get("text"), (String) dict.get("doi"), (String) dict.get("url")));
        }
		this.cite = cites;
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
            tensors.add(new TensorSpecV05((Map<String, Object>) elem, true));
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
            tensors.add(new TensorSpecV05((Map<String, Object>) elem, false));
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
						AxisV05 inAx = (AxisV05) tt.getAxesInfo().getAxesList().stream()
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
				AxisV05 inAx = (AxisV05) this.findInputTensor(ref).getAxesInfo().getAxis(ax.getReferenceAxis());

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
     * @return The ID of this model.
     */
    public String getModelID()
    {
        return modelID;
    }

    /**
     * @return The nickname of this model, for v0.5 is the same as the id.
     */
    public String getNickname()
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
                + ", description=" + description + ", authors=" + authors + ", cite=" + cite
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
	 * Whether the model is already in the local repo or it has to be downloaded
	 * @return true if the model is already installed or false otherwise
	 */
	public boolean isModelInLocalRepo() {
		if (this.localModelPath == null)
			return false;
		return new File(localModelPath).isDirectory();
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
	public String buildInfo() {
		String authorNames = "<ul>";
		for (Author auth : this.authors) {
			authorNames += "<li>" + auth.getName() + "</li>";
		}
		authorNames += "</ul>";
		String citation = "<ul>";
		for (Cite ci : this.cite) {
			if (ci.getUrl() != null && ci.getText() != null)
				citation += "<li><a href='" + ci.getUrl() + "'>" + ci.getText() + "</a></li>";
			else if (ci.getText() != null)
				citation += "<li>" + ci.getText() + "</li>";
		}
		citation += "</ul>";
		if (this.isModelInLocalRepo())
			return String.format(TEXT_DESCRIPTION_LOCAL, this.name, this.getNickname(), 
					this.description, new File(localModelPath).getName(), authorNames, citation);
		else
			return String.format(TEXT_DESCRIPTION, this.name, this.getNickname(), this.description, authorNames, citation);
	}

	@Override
	public String getGitRepo() {
		return this.git_repo;
	}

	@Override
	public List<Badge> getBadges() {
		return new ArrayList<Badge>();
	}

	@Override
	public void addModelPath(Path modelBasePath) {
		this.localModelPath = modelBasePath.toFile().getAbsolutePath();
	}

	@Override
	public String getModelURL() {
		if (this.download_url == null)
			this.download_url = BioimageioRepo.getModelRdfUrl(modelID, version);
		return this.download_url;
	}

	@Override
	public String getRDFSource() {
		return getModelURL() + Constants.RDF_FNAME;
	}
}
