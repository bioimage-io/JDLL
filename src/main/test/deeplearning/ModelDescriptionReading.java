package deeplearning;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.Map;

import org.bioimageanalysis.icy.deeplearning.utils.ModelDescription;
import org.junit.jupiter.api.Test;
import org.yaml.snakeyaml.Yaml;

import lombok.extern.slf4j.Slf4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

@Slf4j
public class ModelDescriptionReading {
    @Test
    public void tryReadYaml() throws FileNotFoundException {
        Yaml yaml = new Yaml();
        InputStream inputStream = new FileInputStream("/Users/ekaterina.moreva/Documents/Embl/model/PlatyTensorflowModel/rdf.yaml");
        Map<String, Object> document = yaml.load(inputStream);
        ModelDescription modelDescription = yaml.load(inputStream);
        assertEquals("README.md",modelDescription.getDocumentation());

//        assertNotNull(document);
//        assertEquals(16, document.size());
    }
}
