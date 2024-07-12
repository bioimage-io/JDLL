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

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import io.bioimage.modelrunner.transformations.PythonTransformation;

/**
 * Class that contains important constants for the software
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class Constants {
	
	final public static String ISSUES_LINK = "https://github.com/bioimage-io/model-runner-java/issues";
	// TODO update wiki link
	final public static String WIKI_LINK = "https://github.com/bioimage-io/model-runner-java/issues";
	final public static String ENGINES_LINK = "https://raw.githubusercontent.com/bioimage-io/JDLL/main/src/main/resources/availableDLVersions.yml";

	/**
	 * File name of the resource description file inside the model folder
	 */
	public static final String RDF_FNAME = "rdf.yaml";
	/**
	 * Last part of files stored in zenodo
	 */
	public static final String ZENODO_ANNOYING_SUFFIX = "/content";
	/**
	 * Zenodo domain
	 */
	public static final String ZENODO_DOMAIN = "https://zenodo.org/";
	/**
	 * Name of the pre- or post-processing that requires using Python
	 */
	public static final String PYTHON_PROCESSING_NAME = PythonTransformation.NAME;
	
	public static final String JDLL_VERSION = getVersion();
	
	public static final String JAR_NAME = getNAME();
	
    private static String getVersion() {
        try (InputStream input = Constants.class.getResourceAsStream(".properties")) {
            Properties prop = new Properties();
            prop.load(input);
            return prop.getProperty("version");
        } catch (IOException ex) {
            return "unknown";
        }
    }
	
    private static String getNAME() {
        try (InputStream input = Constants.class.getResourceAsStream(".properties")) {
            Properties prop = new Properties();
            prop.load(input);
            return prop.getProperty("name");
        } catch (IOException ex) {
            return "unknown";
        }
    }
}
