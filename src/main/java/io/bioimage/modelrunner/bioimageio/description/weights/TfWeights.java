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
package io.bioimage.modelrunner.bioimageio.description.weights;

import java.io.File;
import java.net.MalformedURLException;
import java.util.Map;
import java.util.Set;

import io.bioimage.modelrunner.utils.CommonUtils;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.versionmanagement.SupportedVersions;

/**
 * Class that contains the information for Tensorflow weights.
 * For more information about the parameters go to:
 * https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/weight_formats_spec_0_4.md
 * 
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class TfWeights implements WeightFormat{

	private String compatiblePythonVersion;

	private String weightsFormat;

	private String trainingVersion;

	private String sha256;

	private String source;

	private String parent;
	
	boolean gpu = false;

	private String compatibleVersion;

	/**
	 * Crate an object that specifies Tensorflow weights
	 * 
	 * @param weights
	 * 	part of the yaml file that contains exclusively the 
	 * 	information referring to the Tensorflow weights
	 */
	public TfWeights(Map<String, Object> weights) {
		weightsFormat = ModelWeight.getTensorflowID();
		Set<String> keys = weights.keySet();
		for (String k : keys) {
			Object fieldElement = weights.get(k);
			switch (k)
	        {
            	case "tensorflow_version":
            		setTrainingVersion(fieldElement);
	                break;
	            case "source":
	                setSource(fieldElement);
	                break;
	            case "parent":
	                setParent(fieldElement);
	                break;
	            case "sha256":
	            	setSha256(fieldElement);
	                break;
	            case "architecture":
	            	setArchitecture(fieldElement);
	                break;
	        }
		}
		setCompatibleVersion();
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public String getFramework() {
		return weightsFormat;
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public String getTrainingVersion() {
		return trainingVersion;
	}
	
	/**
	 * Set the training version for the weights
	 * specified in the yaml if it exists
	 *
	 * @param v
	 * 	training version of the weights
	 */
	public void setTrainingVersion(Object v) {
		if (v instanceof String && !((String)v).contains("+")
				 && !((String)v).contains("cu")
				 && !((String)v).contains("cuda"))
			this.trainingVersion = (String) v;
		else if (v instanceof String && ((String)v).contains("+"))
			this.trainingVersion = ((String) v).substring(0, ((String) v).indexOf("+")).trim();
		else if (v instanceof String && ((String)v).contains("cuda"))
			this.trainingVersion = ((String) v).substring(0, ((String) v).indexOf("cuda")).trim();
		else if (v instanceof String && ((String)v).contains("cu"))
			this.trainingVersion = ((String) v).substring(0, ((String) v).indexOf("cu")).trim();
		else if (v instanceof Double)
			this.trainingVersion = "" + v;
		else if (v instanceof Float)
			this.trainingVersion = "" + v;
		else if (v instanceof Long)
			this.trainingVersion = "" + v;
		else if (v instanceof Integer)
			this.trainingVersion = "" + v;
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public String getSha256() {
		return sha256;
	}
	
	/**
	 * Set the SHA256 of the model from the parameters in the yaml
	 *
	 * @param s
	 * 	SHA256 of the model
	 */
	public void setSha256(Object s) {
		if (s instanceof String)
			sha256 = (String) s;
		
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public String getSource() {
		return source;
	}
	
	/**
	 * Set the source of the model from the parameters in the yaml
	 *
	 * @param s
	 * 	string from the yaml file containing the source, return only the
	 * 	name of the file inside the folder, not the whole path
	 */
	public void setSource(Object s) {
		if (s instanceof String)
			this.source = (String) s;
		
	}

	@Override
	public String getParent() {
		return parent;
	}
	
	/**
	 * Set the parent of the weights in the case they exist
	 * @param parent
	 * 	parent weights of the model
	 */
	public void setParent(Object parent) {
		if (parent instanceof String)
			this.parent = (String) parent;
	}

	private ModelArchitecture architecture;
	
	@Override
	/**
	 * {@inheritDoc}
	 */
	public ModelArchitecture getArchitecture() {
		return architecture;
	}
	
	/**
	 * Set the path to the architecture of the weights in the case it exists
	 * @param architecture
	 * 	path to the architecture of the model
	 */
	public void setArchitecture(Object architecture) {
		if (architecture instanceof Map)
			this.architecture = new ModelArchitecture((Map<String, Object>) architecture);
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public String getSourceFileName() {
		if (source == null)
			return source;
		try {
			return CommonUtils.getFileNameFromURLString(source);
		} catch (MalformedURLException e) {
			if (source.startsWith(Constants.ZENODO_DOMAIN) && source.endsWith(Constants.ZENODO_ANNOYING_SUFFIX))
				return new File(source.substring(0, 
						source.length() - Constants.ZENODO_ANNOYING_SUFFIX.length())).getName();
			else
				return new File(source).getName();
		}
	}
	
	/**
	 * {@inheritDoc}
	 * Method to set whether the engine used for this weights supports GPU or not
	 * @param support
	 * 	whether the engine for the weights supports GPu or not
	 */
	@Override
	public void supportGPU(boolean support) {
		gpu = support;
	}
	
	/**
	 * {@inheritDoc}
	 * Method to know whether the engine used for this weights supports GPU or not
	 * @return whether the engine for the weigths supports GPU or not
	 */
	@Override
	public boolean isSupportGPU() {
		return gpu;
	}


	@Override
	/**
	 * {@inheritDoc}
	 */
	public String getJavaTrainingVersion() {
		return compatibleVersion;
	}
	
	/**
	 * Select a version supported by JDLL that is compatible with the training version
	 */
	private void setCompatibleVersion() {
		if (this.trainingVersion == null)
			this.compatibleVersion = null;
		compatibleVersion = SupportedVersions.getJavaVersionForPythonVersion("tensorflow", trainingVersion);
	}

	@Override
	/**
	 * {@inheritDoc}
	 */
	public String getClosestSupportedPythonVersion() {
		if (this.trainingVersion == null)
			return null;
		if (compatiblePythonVersion == null)
			compatiblePythonVersion = SupportedVersions.getClosestSupportedPythonVersion("tensorflow", trainingVersion);
		return compatiblePythonVersion;
	}
}
