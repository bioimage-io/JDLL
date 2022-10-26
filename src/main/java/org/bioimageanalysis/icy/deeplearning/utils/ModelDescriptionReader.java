package org.bioimageanalysis.icy.deeplearning.utils;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.Map;

import org.yaml.snakeyaml.Yaml;

public class ModelDescriptionReader {

    public ModelDescription tryReadYaml(String modelSource) throws FileNotFoundException {
        Yaml yaml = new Yaml();
        InputStream inputStream = new FileInputStream(modelSource);
        Map<String, Object> document = yaml.load(inputStream);
        ModelDescription modelDescription = yaml.load(inputStream);
        return modelDescription;
    }
}
