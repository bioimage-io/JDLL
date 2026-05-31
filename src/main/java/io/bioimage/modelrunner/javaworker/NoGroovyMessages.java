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
package io.bioimage.modelrunner.javaworker;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.reflect.Type;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.ToNumberPolicy;
import com.google.gson.reflect.TypeToken;

/**
 * Gson-backed message utilities for JDLL Java workers.
 * <p>
 * This intentionally mirrors the small JSON subset used by the Java engine
 * workers without depending on Appose's Groovy-backed message implementation.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class NoGroovyMessages {

	public static final String REQUEST_EXECUTE = "EXECUTE";
	public static final String REQUEST_CANCEL = "CANCEL";
	public static final String RESPONSE_LAUNCH = "LAUNCH";
	public static final String RESPONSE_UPDATE = "UPDATE";
	public static final String RESPONSE_COMPLETION = "COMPLETION";
	public static final String RESPONSE_FAILURE = "FAILURE";
	public static final String RESPONSE_CANCELATION = "CANCELATION";

	private static final Gson GSON = new GsonBuilder()
			.setObjectToNumberStrategy(ToNumberPolicy.LONG_OR_DOUBLE)
			.create();
	private static final Type MAP_TYPE = new TypeToken<Map<String, Object>>() {}.getType();

	private NoGroovyMessages() {
		// Utility class.
	}

	/**
	 * Encodes a map as one JSON message.
	 * 
	 * @param data the message map.
	 * @return the encoded JSON.
	 */
	public static String encode(Map<?, ?> data) {
		return GSON.toJson(data);
	}

	/**
	 * Decodes a JSON message.
	 * 
	 * @param json the encoded JSON.
	 * @return the decoded map.
	 */
	public static Map<String, Object> decode(String json) {
		return GSON.fromJson(json, MAP_TYPE);
	}

	/**
	 * Formats a throwable stack trace as a string.
	 * 
	 * @param throwable the throwable.
	 * @return the stack trace string.
	 */
	public static String stackTrace(Throwable throwable) {
		StringWriter writer = new StringWriter();
		throwable.printStackTrace(new PrintWriter(writer));
		return writer.toString();
	}
}
