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
package io.bioimage.modelrunner.bioimageio.download;

import java.io.File;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.download.FileDownloader;
import io.bioimage.modelrunner.engine.installation.EngineInstall;
import io.bioimage.modelrunner.versionmanagement.JarInfo;

/**
 * Class that contains the methods to track the progress downloading files. 
 * The files have to be downloaded to the same folder.
 * It can be used to track the download of any file or list of files. In addition
 * there is a special constructor to track more easily the dowload of Bioimage.io models.
 * 
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class DownloadTracker {
	
	/**
	 * Functional interface to create a consumer that accepts two args and
	 * can be used to retrieve an underlying map
	 * @author Carlos Garcia Lopez de Haro
	 *
	 * @param <T>
	 * 	key 
	 * @param <U>
	 * 	value
	 */
	public static class TwoParameterConsumer<T, U> {
		/**
		 * Map where the values are stored
		 */
		private LinkedHashMap<T, U> map = new LinkedHashMap<T, U>();
		
		/**
		 * Add the key value pair
		 * @param t
		 * 	key
		 * @param u
		 * 	value
		 */
	    public void accept(T t, U u) {
	        map.put(t, u);
	    }
	    
	    /**
	     * Retrieve the map
	     * @return the map
	     */
	    public LinkedHashMap<T, U> get() {
	    	return map;
	    }
	}
	
	/**
	 * Create consumer used to be used with the {@link DownloadTracker}.
	 * This consumer will be where the info about the files downloaded is written.
	 * The key will be the name of the file and the value the size in bytes already
	 * downloaded
	 * @return a consumer to track downloaded files
	 */
	public static TwoParameterConsumer<String, Long> createConsumerTotalBytes() {
		return new TwoParameterConsumer<String, Long>();
	}
	
	/**
	 * Create consumer used to be used with the {@link DownloadTracker}.
	 * This consumer will be where the info about the files downloaded is written.
	 * The key will be the name of the file and the value the porcentage of
	 * the file already downloaded.
	 * @return a consumer to track downloaded files
	 */
	public static TwoParameterConsumer<String, Double> createConsumerProgress() {
		return new TwoParameterConsumer<String, Double>();
	}

}
