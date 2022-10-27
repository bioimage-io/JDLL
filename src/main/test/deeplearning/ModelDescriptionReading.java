package deeplearning;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;

import org.bioimageanalysis.icy.deeplearning.utils.model.description.AuthorsDescription;
import org.bioimageanalysis.icy.deeplearning.utils.model.description.ModelDescription;
import org.junit.jupiter.api.Test;
import org.yaml.snakeyaml.TypeDescription;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.Constructor;

import lombok.extern.slf4j.Slf4j;

import static org.junit.Assert.assertEquals;

@Slf4j
public class ModelDescriptionReading {
    @Test
    public void tryReadYaml() throws FileNotFoundException {
        Constructor constructor = new Constructor(ModelDescription.class);
        TypeDescription customTypeDescription = new TypeDescription(ModelDescription.class);
        customTypeDescription.addPropertyParameters("authors", AuthorsDescription.class);
        constructor.addTypeDescription(customTypeDescription);
        Yaml yaml = new Yaml(constructor);
        InputStream inputStream = new FileInputStream("/Users/ekaterina.moreva/Documents/Embl/model/PlatyTensorflowModel/rdf.yaml");

        ModelDescription modelDescription = yaml.load(inputStream);
        assertEquals("README.md", modelDescription.getDocumentation());

    }
}
