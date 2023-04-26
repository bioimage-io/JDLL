package io.bioimage.modelrunner.bioimageio.description.weights;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** TODO make more stable. Talk to Wei to define it better
 * Class that contains the information for the Bioengine weights.
 * For more information about the parameters go to:
 * https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/weight_formats_spec_0_4.md
 * 
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class BioEngineWeights implements WeightFormat{

	/**
	 * Crate an object that specifies BioEngine weights
	 * 
	 * @param server
	 * 	String to the server where the BioEngine is going to be called from
	 * @throws MalformedURLException if the String does not correspond to a valid URL
	 */
	public BioEngineWeights(String server) throws MalformedURLException {
		weightsFormat = "bioengine";
		setSource(server);
	}

	private String weightsFormat;
	@Override
	public String getWeightsFormat() {
		return weightsFormat;
	}

	private String trainingVersion = "";
	@Override
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
		if (v instanceof String)
			this.trainingVersion = (String) v;
		else if (v instanceof Double)
			this.trainingVersion = "" + v;
		else if (v instanceof Float)
			this.trainingVersion = "" + v;
		else if (v instanceof Long)
			this.trainingVersion = "" + v;
		else if (v instanceof Integer)
			this.trainingVersion = "" + v;
	}

	private String sha256 = "";
	@Override
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

	private String source;
	@Override
	public String getSource() {
		return source;
	}
	
	/**
	 * Set the source of the model from the parameters in the yaml
	 *
	 * @param s
	 * 	string from the yaml file containing the source, return only the
	 * 	name of the file inside the folder, not the whole path
	 * @throws MalformedURLException if the String does not correspond to a valid URL
	 */
	public void setSource(Object s) throws MalformedURLException {
		if (s instanceof String) {
			this.source = (String) s;
			new URL(this.source);
		}
		
	}

	private List<String> authors = new ArrayList<String>();
	@Override
	public List<String> getAuthors() {
		return authors;
	}
	
	/**
	 * Set the authors of the model
	 * @param authors
	 * 	authors of the model
	 */
	public void setAuthors(Object authors) {
		if (authors instanceof String) {
			List<String> authList = new ArrayList<String>();
			authList.add((String) authors);
			this.authors = authList;
		} else if (authors instanceof List<?>) {
			this.authors = (List<String>) authors;
		}
		
	}

	private Map<String, Object> attachments;
	@Override
	public Map<String, Object> getAttachments() {
		return attachments;
	}
	
	/**
	 * Set the attachments of the weights if they exist
	 * @param attachments
	 * 	attachments of the model
	 */
	public void setAttachments(Object attachments) {
		if (attachments instanceof Map<?, ?>)
			this.attachments = (Map<String, Object>) attachments;
		
	}

	private String parent;
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

	private String architecture;
	@Override
	public String getArchitecture() {
		return architecture;
	}
	
	/**
	 * Set the path to the architecture of the weights in the case it exists
	 * @param architecture
	 * 	path to the architecture of the model
	 */
	public void setArchitecture(Object architecture) {
		if (architecture instanceof String)
			this.architecture = (String) architecture;
	}

	private String architectureSha256;
	@Override
	public String getArchitectureSha256() {
		return architectureSha256;
	}
	
	/**
	 * Set the architecture Sha256 in the case it exists
	 * @param architectureSha256
	 * 	architecture Sha256 of the model
	 */
	public void setArchitectureSha256(Object architectureSha256) {
		if (architectureSha256 instanceof String)
			this.architectureSha256 = (String) architectureSha256;
	}

	@Override
	public String getSourceFileName() {
		if (source == null)
			return source;
		return new File(source).getName();
	}
	
	boolean gpu = true;
	/**
	 * Method to set whether the engine used for this weights supports GPU or not.
	 * However, the server always uses GPU, and the BioEngine weights always run
	 * on a server. This method is not relevant for the BioEngine weights
	 * @return
	 */
	@Override
	public void supportGPU(boolean support) {
		gpu = support;
	}
	
	/**
	 * Method to know whether the engine used for this weights supports GPU or 
	 * However, the server always uses GPU, and the BioEngine weights always run
	 * on a server. This method is not relevant for the BioEngine weights
	 * @return
	 */
	@Override
	public boolean isSupportGPU() {
		return true;
	}
}
