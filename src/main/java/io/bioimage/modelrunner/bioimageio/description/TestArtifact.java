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
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import io.bioimage.modelrunner.utils.Constants;

/**
 * Define each of the test inputs and outputs defined in the rdf.yaml file
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class TestArtifact {
	/**
	 * String to the sample image
	 */
	private String string;
	/**
	 * URL for the sample image
	 */
	private URL url;
	/**
	 * Path to the sample image in the local machine
	 */
	private Path path;
	/**
	 * List of allowed extensions for the sample images
	 */
	private static List<String> allowedExtensions = Arrays.asList(new String[] {".npy"});
	
    /**
     * Creates a {@link TestArtifact} instance.
     * 
     * @param sampleInputUrl
     * 	String specified in the yaml file as a sample input or output
     * @return The creates instance.
     */
    public static TestArtifact build(String sampleInputUrl)
    {
    	TestArtifact sampleInput = new TestArtifact();
        if (sampleInputUrl == null)
        	return null;
        sampleInput.string = sampleInputUrl;
        if (!sampleInput.isExtensionAllowed())
        	return null;
        sampleInput.createSampleInputURL();
        sampleInput.createSampleInputPath();
        sampleInput.createSampleInputPath();
        return sampleInput;        
    }
    
    /**
     * Check if the extensio of the sample image is allowed or not
     * @return true if the sample image contains a valid extension, false otherwise
     */
    public boolean isExtensionAllowed() {
    	if (getFileExtension() == null)
    		return false;
    	if (!allowedExtensions.contains(this.getFileExtension().toLowerCase()))
    		return false;
    	return true;
    }
    
    /**
     * Get the extension of the file containing the sample image
     * @return the extension of teh sample image file
     */
    public String getFileExtension() {
    	if (string == null)
    		return null;
    	if (string.lastIndexOf(".") == -1)
    		return null;
    	if (string.startsWith(Constants.ZENODO_DOMAIN) && string.endsWith(Constants.ZENODO_ANNOYING_SUFFIX))
    		return string.substring(string.lastIndexOf("."), string.length() - Constants.ZENODO_ANNOYING_SUFFIX.length());
    	return string.substring(string.lastIndexOf("."));
    }
    
    /**
     * Try to create an URL for the sample input if it is specified as an URL
     */
    private void createSampleInputURL() {
		try {
			url = new URL(string);
		} catch (MalformedURLException e) {
		}
    }
    
    /**
     * Create the Path to a sample input if it is defined in the yaml file as a local file
     */
    private void createSampleInputPath() {
    	try {
	    	path = Paths.get(string);
	    	if (!path.toFile().exists())
	    		path = null;
    	} catch (Exception ex) {
    	}
    }
    
    /**
     * Add the local model path to the sample images to know where these images are
     * @param p
     * 	the path to the local model folder
     */
    public void addLocalModelPath(Path p) {
    	if (!p.toFile().exists())
    		return;
    	String name = new File(string).getName();
    	Path nPath = p.resolve(name);
    	if (nPath.toFile().exists())
    		path = nPath;
    }
    
    /**
     * Return the String URL of the sample image
     * @return the string URL of the sample image
     */
    public String getString() {
    	return string;
    }
    
    /**
     * Return the url of the sample image
     * @return the url corresponding where the sample image is in the cloud
     */
    public URL getUrl() {
    	return url;
    }
    
    /**
     * Return the local path to the sample image
     * @return the local path to the sample image
     */
    public Path getLocalPath() {
    	return path;
    }

	@Override
    public String toString()
    {
		String str = "TestNpy {";
		str += " string=" + string;
		if (url != null)
			str += " url=" + url.toString();
		if (path != null)
			str += " path=" + path.toString();
		str += " }";
        return str;
    }
	
	/**
	 * Return list of allowed extensions
	 * @return the allowed extensions for sample images
	 */
	public static List<String> getAllowedExtensions(){
		return allowedExtensions;
	}

}
