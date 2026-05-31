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
package io.bioimage.modelrunner.model.special.common;

import java.lang.reflect.Array;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Shared helpers for model-specific training code that is executed through Appose.
 */
public final class TrainingCodeUtils {

	private TrainingCodeUtils() {
		// Utility class.
	}

	/**
	 * Forwards Appose debug lines except raw task updates, which are already handled through task listeners.
	 *
	 * @param line the line.
	 * @param logConsumer the log consumer callback.
	 */
	public static void logTrainingDebug(String line, Consumer<String> logConsumer) {
		if (logConsumer == null || isTaskUpdateDebugLine(line)) {
			return;
		}
		logConsumer.accept(line);
	}

	private static boolean isTaskUpdateDebugLine(String line) {
		return line != null && line.contains("\"responseType\":\"UPDATE\"");
	}

	/**
	 * Converts a numeric event value to an int.
	 *
	 * @param value the value.
	 * @param fallback the fallback.
	 * @return the resulting int.
	 */
	public static int asInt(Object value, int fallback) {
		return value instanceof Number ? ((Number) value).intValue() : fallback;
	}

	/**
	 * Converts a protocol object into an ordered double map.
	 *
	 * @param value the value.
	 * @return the resulting map.
	 */
	public static Map<String, Double> asDoubleMap(Object value) {
		Map<String, Double> result = new LinkedHashMap<String, Double>();
		if (!(value instanceof Map)) {
			return result;
		}
		for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
			if (entry.getKey() != null && entry.getValue() instanceof Number) {
				result.put(entry.getKey().toString(), ((Number) entry.getValue()).doubleValue());
			}
		}
		return result;
	}

	/**
	 * Escapes a Java string for insertion inside a Python raw string literal delimited by single quotes.
	 *
	 * @param value the value.
	 * @return the resulting string.
	 */
	public static String py(String value) {
		return value == null ? "" : value.replace("\\", "\\\\").replace("'", "\\'");
	}

	/**
	 * Minimal JSON encoder for the simple config structures used in generated training scripts.
	 *
	 * @param value the value.
	 * @return the resulting string.
	 */
	public static String toJson(Object value) {
		if (value == null) {
			return "null";
		}
		if (value instanceof String) {
			return "\"" + ((String) value).replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
		}
		if (value instanceof Boolean || value instanceof Number) {
			if (value instanceof Double && !Double.isFinite(((Double) value).doubleValue())) {
				return "null";
			}
			if (value instanceof Float && !Float.isFinite(((Float) value).floatValue())) {
				return "null";
			}
			return String.format(Locale.ROOT, "%s", value);
		}
		if (value instanceof Map) {
			StringBuilder builder = new StringBuilder("{");
			boolean first = true;
			for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
				if (!first) {
					builder.append(',');
				}
				first = false;
				builder.append(toJson(String.valueOf(entry.getKey()))).append(':').append(toJson(entry.getValue()));
			}
			return builder.append('}').toString();
		}
		if (value instanceof Iterable) {
			StringBuilder builder = new StringBuilder("[");
			boolean first = true;
			for (Object item : (Iterable<?>) value) {
				if (!first) {
					builder.append(',');
				}
				first = false;
				builder.append(toJson(item));
			}
			return builder.append(']').toString();
		}
		if (value.getClass().isArray()) {
			StringBuilder builder = new StringBuilder("[");
			int length = Array.getLength(value);
			for (int i = 0; i < length; i++) {
				if (i > 0) {
					builder.append(',');
				}
				builder.append(toJson(Array.get(value, i)));
			}
			return builder.append(']').toString();
		}
		return toJson(String.valueOf(value));
	}

	/**
	 * Captures Appose's original stdout before user logs are redirected.
	 *
	 * @return Python source
	 */
	public static String apposeStdoutCapture() {
		return "_appose_stdout = sys.stdout" + System.lineSeparator();
	}

	/**
	 * Emits an Appose-safe {@code task.update} wrapper that temporarily restores Appose stdout.
	 *
	 * @param functionName the function name.
	 * @return the resulting string.
	 */
	public static String taskUpdateFunction(String functionName) {
		String nl = System.lineSeparator();
		return ""
				+ "def " + functionName + "(**kwargs):" + nl
				+ "  current_stdout = sys.stdout" + nl
				+ "  try:" + nl
				+ "    sys.stdout = _appose_stdout" + nl
				+ "    task.update(**kwargs)" + nl
				+ "  finally:" + nl
				+ "    sys.stdout = current_stdout" + nl;
	}

	/**
	 * Emits a Python scalar converter for training callbacks.
	 *
	 * @param functionName the function name.
	 * @param detachTensor the detach tensor.
	 * @return the resulting string.
	 */
	public static String scalarFunction(String functionName, boolean detachTensor) {
		String nl = System.lineSeparator();
		String detachCode = detachTensor
				? "    if hasattr(value, 'detach'):" + nl
						+ "      value = value.detach().cpu()" + nl
				: "    if value is None:" + nl
						+ "      return None" + nl;
		return ""
				+ "def " + functionName + "(value):" + nl
				+ "  try:" + nl
				+ detachCode
				+ "    if hasattr(value, 'item'):" + nl
				+ "      value = value.item()" + nl
				+ "    return float(value)" + nl
				+ "  except Exception:" + nl
				+ "    return None" + nl;
	}

	/**
	 * Emits a Python dict cleaner that keeps only scalar numeric values.
	 *
	 * @param functionName the function name.
	 * @param scalarFunctionName the scalar function name.
	 * @return the resulting string.
	 */
	public static String cleanDictFunction(String functionName, String scalarFunctionName) {
		String nl = System.lineSeparator();
		return ""
				+ "def " + functionName + "(values):" + nl
				+ "  out = {}" + nl
				+ "  if values is None:" + nl
				+ "    return out" + nl
				+ "  for k, v in dict(values).items():" + nl
				+ "    sv = " + scalarFunctionName + "(v)" + nl
				+ "    if sv is not None:" + nl
				+ "      out[str(k)] = sv" + nl
				+ "  return out" + nl;
	}
}
