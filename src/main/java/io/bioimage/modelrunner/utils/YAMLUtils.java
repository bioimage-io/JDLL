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
package io.bioimage.modelrunner.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.yaml.snakeyaml.Yaml;

/**
 * Utility class for YAML files and data structures.
 * 
 * @author Daniel Felipe Gonzalez Obando
 */
public class YAMLUtils
{

    /**
     * Reads the provided yaml file and loads it into a map of string keys and object values.
     * 
     * @param yamlFile
     *        The target yaml file.
     * @return The map loaded with the yaml elements.
     * @throws FileNotFoundException
     *         If the file cannot be found.
     * @throws IOException
     *         If the file cannot be read.
     */
    public static Map<String, Object> load(String yamlFile) throws FileNotFoundException, IOException
    {
        File initialFile = new File(yamlFile);
        try (InputStream targetStream = new FileInputStream(initialFile))
        {
            Yaml yaml = new Yaml();
            Map<String, Object> obj = yaml.load(targetStream);
            return obj;
        }
    }

    /**
     * Reads the provided yaml String and loads it into a map of string keys and object values.
     * 
     * @param yamlString
     *        The String yaml file.
     * @return The map loaded with the yaml elements.
     */
    public static Map<String, Object> loadFromString(String yamlString)
    {
    	Yaml yaml = new Yaml();
    	HashMap<String,Object> yamlElements = yaml.load(yamlString);
        return yamlElements;
    }

    /**
     * Converts a list of objects into an array of strings.
     * 
     * @param list
     *        The target list.
     * @return The array of strings.
     */
    public static String[] castListToStringArray(List<?> list)
    {
        String[] array = new String[list.size()];
        int c = 0;
        for (Object in : list)
        {
            array[c++] = (String) in;
        }
        return array;
    }

    /**
     * Converts a list of objects into an array of integers.
     * 
     * @param list
     *        The target list.
     * @return The array of integers.
     */
    public static int[] castListToIntArray(List<?> list)
    {
        int[] array = new int[list.size()];
        int c = 0;
        for (Object in : list)
        {
            array[c++] = Integer.parseInt(in.toString());
        }
        return array;
    }

    /**
     * Converts a list of objects into an array of doubles.
     * 
     * @param list
     *        The target list.
     * @return The array of doubles.
     */
    public static double[] castListToDoubleArray(List<?> list)
    {
        try
        {
            double[] array = new double[list.size()];
            int c = 0;
            for (Object in : list)
            {
                array[c++] = Double.parseDouble(in.toString());
            }
            return array;
        }
        catch (Exception ex)
        {
            return null;
        }
    }

    /**
     * Converts a list of objects into an array of floats.
     * 
     * @param list
     *        The target list.
     * @return The array of floats.
     */
    public static float[] castListToFloatArray(List<?> list)
    {
        try
        {
            float[] array = new float[list.size()];
            int c = 0;
            for (Object in : list)
            {
            	if (in instanceof String) {
                    array[c++] = Float.parseFloat((String) in);
            	} else if (in instanceof Integer) {
                    array[c++] = ((Integer) in).floatValue();
            	} else if (in instanceof Double) {
                    array[c++] = ((Double) in).floatValue();
            	} else if (in instanceof Float) {
                    array[c++] = ((Float) in).floatValue();
            	} else if (in instanceof Long) {
                    array[c++] = ((Long) in).floatValue();
            	} else if (in instanceof Number) {
                    array[c++] = ((Number) in).floatValue();
            	} else if (in instanceof Byte) {
                    array[c++] = ((Byte) in).floatValue();
            	} else {
            		throw new IllegalArgumentException("Unsupported type for list of Objects:" + in.getClass());
            	}
            }
            return array;
        }
        catch (ClassCastException ex)
        {
            return null;
        }
    }

}
