/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.apposed.appose;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
public final class Types {

	private Types() {
		// NB: Prevent instantiation of utility class.
	}

	/**
	 * Encode a Map into a String
	 * @param data
	 * 	data that wants to be encoded
	 * @return string containing the info of the data map
	 */
	public static String encode(Map<?, ?> data) {
		Gson gson = new Gson();
		return gson.toJson(data);
	}

	@SuppressWarnings("unchecked")
	/**
	 * Convert a json string into a map
	 * @param json
	 * 	json string
	 * @return a map of with the information of the json
	 */
	public static Map<String, Object> decode(String json) {
		Gson gson = new Gson();
        Type mapType = new TypeToken<HashMap<String, Object>>() {}.getType();
        return gson.fromJson(json, mapType);
	}

	/** Dumps the given exception, including stack trace, to a string. 
	 * 
	 * @param t
	 * 	the given exception {@link Throwable}
	 * @return the String containing the whole exception trace
	 */
	public static String stackTrace(Throwable t) {
		StringWriter sw = new StringWriter();
		t.printStackTrace(new PrintWriter(sw));
		return sw.toString();
	}
}
