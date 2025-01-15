package io.bioimage.modelrunner.versionmanagement;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URL;
import java.util.Map;

public class JarInfo {

    // Static instance for the Singleton
    private static JarInfo instance;

    // Map to store the parsed JSON data
    private Map<String, Integer> urlData;
        
    private static URL FILE_PATH = JarInfo.class.getClassLoader().getResource("jar_sizes.json");

    // Private constructor to restrict instantiation
    private JarInfo() throws IOException {
        loadJsonData();
    }

    /**
     * Public method to initialize or get the singleton instance.
     *
     * @return The single instance of {@link JarInfo}
     * @throws IOException If the file cannot be loaded
     */
    public static JarInfo getInstance() throws IOException {
        if (instance == null) {
            instance = new JarInfo();
        }
        return instance;
    }

    /**
     * Public method to access the URL data
     *
     * @return Map containing the URL and their respective sizes
     */
    public Map<String, Integer> getAllData() {
        return urlData;
    }

    /**
     * Method to retrieve the size for a specific URL
     *
     * @param url The URL to look up
     * @return The size associated with the URL, or null if not found
     */
    public Integer get(String url) {
        return urlData.get(url);
    }

    // Private method to load and parse the JSON data
    @SuppressWarnings("unchecked")
	private void loadJsonData() throws IOException {
        ObjectMapper objectMapper = new ObjectMapper();
        urlData = objectMapper.readValue(FILE_PATH, Map.class);
    }
}
