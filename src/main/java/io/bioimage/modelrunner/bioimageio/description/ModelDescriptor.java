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

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;


/**
 * A data structure holding a single Bioimage.io pretrained model description. This instances are created by opening a {@code model.yaml} file.
 * More info about the parameters can be found at:
 * https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/model_spec_latest.md
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public interface ModelDescriptor {
    
    /**
     * Create a set of specifications about the basic info of the model: name od the model, authors,
     * references and Deep Learning framework
     * @return a set of specs for the model
     */
    public String buildBasicInfo();
    
    /**
     * Write the tiling specs for the model
     * @return the tiling specs for the model
     */
    public String buildTilingInfo();
    
    /**
     * Create specifications of the model containing the most important
     * info that is going to be displayed on the DeepIcy plugin
     * @return a String with the most important info
     */
    public String buildInfo();

    /**
     * @return The version of the format used in the descriptor file.
     */
    public String getFormatVersion();

    /**
     * @return The name of this model.
     */
    public String getName();

    /**
     * @return The ID of this model.
     */
    public String getModelID();

    /**
     * @return The ID of this model.
     */
    public String getNickname();

    /**
     * @return The creation timestamp of this model.
     */
    public String getTimestamp();

    /**
     * @return The description of this model.
     */
    public String getDescription();

    /**
     * @return The list of authors for this model.
     */
    public List<Author> getAuthors();

    /**
     * @return The list of citations for this model.
     */
    public List<Cite> getCite();

    /**
     * @return The URL of the git repository of this model.
     */
    public String getGitRepo();

    /**
     * @return The list of tags associated with this model.
     */
    public List<String> getTags();

    /**
     * @return The license description for this model.
     */
    public String getLicense();

    /**
     * @return the type of Bioimage.io artifact that the rdf.yaml
     * refers to. It can be model, dataset, application...
     */
    public String getType();

    /**
     * @return The documentation text associated to this model.
     */
    public String getDocumentation();

    /**
     * @return The list of URIs of the covers for this model.
     */
    public List<String> getCovers();

    /**
     * @return The list of input tensor specification instances for this model.
     */
    public List<TensorSpec> getInputTensors();

    /**
     * Searches for an input tensor with the given name.
     * 
     * @param name
     *        Name of the tensor.
     * @return The tensor with the provided name. null is returned if no tensor is found or if the input tensors list is not initialized.
     */
    public TensorSpec findInputTensor(String name);

    /**
     * Searches for an output tensor with the given name.
     * 
     * @param name
     *        Name of the tensor.
     * @return The tensor with the provided name. null is returned if no tensor is found or if the output tensors list is not initialized.
     */
    public TensorSpec findOutputTensor(String name);

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
    public List<TensorSpec> getOutputTensors();

    /**
     * @return The execution configuration instance for this model.
     */
    public ExecutionConfig getConfig();

    /**
     * @return The model weights instance for this model.
     */
    public ModelWeight getWeights();

	/**
	 * @return the maintainers
	 */
	public List<Author> getMaintainers();

	/**
	 * @return the packaged_by
	 */
	public List<Author> getPackagedBy();

	/**
	 * @return the badges
	 */
	public List<Badge> getBadges();

	/**
	 * @return the attachments
	 */
	public Map<String, Object> getAttachments();

	/**
	 * @return the rdf_source
	 */
	public String getRDFSource();

	/**
	 * @return the version
	 */
	public String getVersion();

	/**
	 * @return the links
	 */
	public List<String> getLinks();
	
	/**
	 * Whether the model is already in the local repo or it has to be downloaded
	 * @return true if the model is already installed or false otherwise
	 */
	public boolean isModelInLocalRepo();
	
	/**
	 * Add the path where the local model is stored to the model descriptor
	 * @param modelBasePath
	 * 	the path to the model in the local machine
	 */
	public void addModelPath(Path modelBasePath);
	
	/**
	 * Return String to path where the model is stored
	 * @return String directory where the model is stored
	 */
	public String getModelPath();
	
	/**
	 * Method that returns whether tiling is allowed or not for the model
	 * @return true if tiling is allowed and false otherwise
	 */
	public boolean isTilingAllowed();
	
	/**
	 * Method that returns whether the model is pyramidal or not
	 * @return true if the model is pyramidal, false otherwise
	 */
	public boolean isPyramidal();
	
	/**
	 * 
	 * @return whether the model can be run on the bioengino or not
	 */
	public boolean canRunOnBioengine();

	public String getModelURL();
}
