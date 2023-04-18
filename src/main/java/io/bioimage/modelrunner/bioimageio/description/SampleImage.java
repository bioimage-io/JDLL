package io.bioimage.modelrunner.bioimageio.description;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/**
 * Define each of the sample inputs spedified in the yaml file
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class SampleImage {
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
	private static List<String> allowedExtensions = Arrays.asList(new String[] {".tiff", ".tif",
			".png", ".jpg", ".jpeg", ".gif"});
	
    /**
     * Creates a {@link SampleImage} instance.
     * 
     * @param sampleInputUrl
     * 	String specified in the yaml file as a sample input or output
     * @return The creates instance.
     */
    public static SampleImage build(String sampleInputUrl)
    {
    	SampleImage sampleInput = new SampleImage();
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
     * @return
     */
    public String getFileExtension() {
    	if (string == null)
    		return null;
    	if (string.lastIndexOf(".") == -1)
    		return null;
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
     * Return the String of the sample image
     * @return
     */
    public String getString() {
    	return string;
    }
    
    /**
     * Return the url of the sample image
     * @return
     */
    public URL getUrl() {
    	return url;
    }
    
    /**
     * Return the url of the sample url
     * @return
     */
    public Path getPath() {
    	return path;
    }

	@Override
    public String toString()
    {
		String str = "SampleInput {";
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
	 * @return
	 */
	public static List<String> getAllowedExtensions(){
		return allowedExtensions;
	}

}