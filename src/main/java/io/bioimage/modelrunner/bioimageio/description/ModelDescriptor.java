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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.utils.Constants;


/**
 * A data structure holding a single Bioimage.io pretrained model description. This instances are created by opening a {@code model.yaml} file.
 * More info about the parameters can be found at:
 * https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/model_spec_latest.md
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public abstract class ModelDescriptor {
	
	protected String format_version;
    protected String name;
    protected String download_url;
    protected String timestamp;
    protected String description;
    protected String type;
    protected String git_repo;
    protected List<Author> authors;
    protected List<Author> maintainers;
    protected List<Author> packaged_by;
    protected List<Cite> cite;
    protected List<String> tags;
    protected String license;
    protected String documentation;
    protected List<Badge> badges;
    protected List<String> covers;
    protected List<TensorSpec> input_tensors;
    protected List<TensorSpec> output_tensors;
    protected ExecutionConfig config;
    protected ModelWeight weights;
    protected Map<String, Object> attachments;
    protected String version;
    protected List<String> links;
    protected static String fromLocalKey = "fromLocalRepo";
    protected static String modelPathKey = "modelPath";
    protected String modelID;
    protected String localModelPath;
    protected boolean supportBioengine = false;
	protected  Map<String, Object> yamlElements;

	protected static final String TEXT_DESCRIPTION = "<html><body>"
		    + "<h3>%s</h3>"
		    + "<p><strong>Nickname:</strong> %s</p>"
		    + "<p><strong>Description:</strong> %s</p>"
		    + "<p><strong>Author(s):</strong></p>"
		    + "%s"
		    + "<p><strong>Citation:</strong></p>"
		    + "%s"
		    + "</body></html>";
	protected static final String TEXT_DESCRIPTION_LOCAL = "<html><body>"
		    + "<h3>%s</h3>"
		    + "<p><strong>Nickname:</strong> %s</p>"
		    + "<p><strong>Description:</strong> %s</p>"
		    + "<p><strong>Folder name:</strong> %s</p>"
		    + "<p><strong>Author(s):</strong></p>"
		    + "%s"
		    + "<p><strong>Citation:</strong></p>"
		    + "%s"
		    + "</body></html>";


    /**
     * @return The ID of this model.
     */
    public abstract String getNickname();
    
    protected abstract List<TensorSpec> buildInputTensors();
    
    protected abstract List<TensorSpec> buildOutputTensors();

    /**
     * Calculate the total input halo once the output tensors are set.
     * NOte that the total halo is calculated following "xyczb" axes order,
     * not the input axes order, as for several inputs the axes order might change
     * for each of them
     * @return the total input halo in "xyczb" axes order
     */
    protected abstract void calculateTotalInputHalo();
    
    protected abstract String findID();
    
    protected abstract void addBioEngine();
    
    /**
     * Build a {@link ModelDescriptorV05} object from a map containing the elements read from
     * a rdf.yaml file
     * @param yamlElements
     * 	map with the information read from a yaml file
     * @return a {@link ModelDescriptorV05} with the info of a Bioimage.io model
     * @throws ModelSpecsException if any of the parameters in the rdf.yaml file does not make fit the constraints
     */
    protected void buildModelDescription()
    {

        Set<String> yamlFields = yamlElements.keySet();
        String[] yamlFieldsArr = new String[yamlFields.size()];
        Arrays.sort(yamlFields.toArray(yamlFieldsArr));
        for (String field : yamlFieldsArr)
        {
                switch (field)
                {
                    case "format_version":
                        format_version = (String) yamlElements.get(field);
                        break;
                    case "name":
                        name = (String) yamlElements.get(field);
                        break;
                    case "timestamp":
                        timestamp = yamlElements.get(field).toString();
                        break;
                    case "description":
                        description = (String) yamlElements.get(field);
                        break;
                    case "id":
                        modelID = (String) yamlElements.get(field);
                        break;
                    case "authors":
                        this.authors = buildAuthors(yamlElements.get(field));
                        break;
                    case "maintainers":
                        this.maintainers = buildAuthors(yamlElements.get(field));
                        break;
                    case "packaged_by":
                    	this.packaged_by = buildAuthors(yamlElements.get(field));
                        break;
                    case "tags":
                        tags = castListStrings(yamlElements.get(field));
                        break;
                    case "links":
                        links = castListStrings(yamlElements.get(field));
                        break;
                    case "license":
                        license = (String) yamlElements.get(field);
                        break;
                    case "documentation":
                        documentation = (String) yamlElements.get(field);
                        break;
                    case "git_repo":
                    	git_repo = (String) yamlElements.get(field);
                        break;
                    case "type":
                        type = (String) yamlElements.get(field);
                        break;
                    case "attachments":
                        // TODO createAttachments();
                        break;
                    case "covers":
                    	covers = castListStrings(yamlElements.get(field));
                        break;
                    case "modelPath":
                        localModelPath = (String) yamlElements.get(field);
                        break;
                    default:
                        break;
            }
        }
        cite = buildCiteElements();
        input_tensors = this.buildInputTensors();
        output_tensors = this.buildOutputTensors();
        weights = this.buildWeights();
        config = this.buildConfig();
        calculateTotalInputHalo();
        
        addBioEngine();
        if (localModelPath != null)
        	addModelPath(new File(localModelPath).toPath());
    }

	protected List<Author> buildAuthors(Object object) {
		List<Author> authors = new ArrayList<>();
    	if (object == null || !(object instanceof List)) {
            this.authors = authors;
            return authors;
    	}
        for (Object elem : (List<Object>) object)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            @SuppressWarnings("unchecked")
            Map<String, String> dict = (Map<String, String>) elem;
            authors.add(Author.build(dict.get("affiliation"), dict.get("email"), dict.get("github_user"), dict.get("name"), dict.get("orcid")));
        }
        return authors;
	}

	protected List<Cite> buildCiteElements() {
		Object citeElements = this.yamlElements.get("cite");
        List<Cite> cites = new ArrayList<Cite>();
    	if (citeElements == null || !(citeElements instanceof List<?>)) {
    		this.cite = cites;
    		return cites;
    	}
        for (Object elem : (List) citeElements)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            @SuppressWarnings("unchecked")
            Map<String, Object> dict = (Map<String, Object>) elem;
            cites.add(Cite.build((String) dict.get("text"), (String) dict.get("doi"), (String) dict.get("url")));
        }
		return cites;
	}

	protected ModelWeight buildWeights() {
        return ModelWeight.build((Map<String, Object>) this.yamlElements.get("weights"));
	}
	
	protected ExecutionConfig buildConfig() {
        return ExecutionConfig.build((Map<String, Object>) this.yamlElements.get("config"));
	}
	
    /**
     * Create a set of specifications about the basic info of the model: name od the model, authors,
     * references and Deep Learning framework
     * @return a set of specs for the model
     */
    public String buildInfo() {
		String authorNames = "<ul>";
		for (Author auth : this.authors) {
			authorNames += "<li>" + auth.getName() + "</li>";
		}
		authorNames += "</ul>";
		String citation = "<ul>";
		if (this.cite == null)
			cite = new ArrayList<>();
		for (Cite ci : this.cite) {
			if (ci.getUrl() != null && ci.getText() != null)
				citation += "<li><a href='" + ci.getUrl() + "'>" + ci.getText() + "</a></li>";
			else if (ci.getText() != null)
				citation += "<li>" + ci.getText() + "</li>";
		}
		citation += "</ul>";
		if (this.isModelInLocalRepo())
			return String.format(ModelDescriptor.TEXT_DESCRIPTION_LOCAL, this.name, this.getNickname(), 
					this.description, new File(localModelPath).getName(), authorNames, citation);
		else
			return String.format(ModelDescriptor.TEXT_DESCRIPTION, this.name, this.getNickname(), this.description, authorNames, citation);
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
     * @return The URL of the git repository of this model.
     */
    public String getGitRepo()
    {
        return git_repo;
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
	 * @return the rdf_source
	 */
    public String getRDFSource() {
		return getModelURL() + Constants.RDF_FNAME;
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
	 * Return String to path where the model is stored
	 * @return String directory where the model is stored
	 */
	public void addModelPath(Path modelBasePath) {
		this.localModelPath = modelBasePath.toFile().getAbsolutePath();
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
	 * 
	 * @return the url to the online folder that contains all the files of the model
	 */
	public String getModelURL() {
		if (this.download_url == null)
			this.download_url = BioimageioRepo.getModelRdfUrl(modelID, version);
		return this.download_url;
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

	@Override
    public String toString()
    {
        return "ModelDescription {formatVersion=" + format_version + ", name=" + name + ", timestamp=" + timestamp
                + ", description=" + description + ", authors=" + authors + ", cite=" + cite + ", gitRepo=" + git_repo
                + ", tags=" + tags + ", license=" + license + ", documentation=" + documentation + ", covers=" + covers
                + ", inputTensors=" + input_tensors + ", outputTensors=" + output_tensors + ", config=" + config
                + ", weights=" + weights + "}";
    }
}
