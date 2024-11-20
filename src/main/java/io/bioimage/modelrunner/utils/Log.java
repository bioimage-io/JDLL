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
package io.bioimage.modelrunner.utils;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.function.Consumer;

public class Log {
	/**
	 * Format of the time
	 */
	public static SimpleDateFormat format = new SimpleDateFormat("HH:mm:ss");

	/**
	 * Get current time.
	 * REturns the time of the day in "HH:mm:ss" format with military time.
	 * Example 16:20:59
	 * @return the time
	 */
	public static String gct() {
		return format.format(new Date());
	}
	
	/**
	 * Write the wanted string to a Consumer to communicate with the main interface.
	 * It also prints the message on the terminal
	 * @param consumer
	 * 	consumer to communicate with the main code. This parameter can be null and 
	 * 	then the log will only be displayed on the terminal.
	 * @param str
	 * 	String to communicate
	 */
	public static void addProgressAndShowInTerminal(Consumer<String> consumer, String str) {
		addProgressAndShowInTerminal(consumer, str, false);
	}
	
	/**
	 * Write the wanted string to a Consumer to communicate with the main interface.
	 * It also prints the message on the terminal
	 * @param consumer
	 * 	consumer to communicate with the main code. This parameter can be null and 
	 * 	then the log will only be displayed on the terminal.
	 * @param str
	 * 	String to communicate
	 * @param addTime
	 * 	whether to specify the time along with the message or not
	 */
	public static void addProgressAndShowInTerminal(Consumer<String> consumer, String str, boolean addTime) {
		if (addTime)
			str = Log.gct() + " -- " + str + System.lineSeparator();
		else
			str += System.lineSeparator();
		if (consumer != null)
			consumer.accept(str);
		System.out.println(str);
	}
	
	/**
	 * Write the wanted string to a Consumer to communicate with the main interface.
	 * @param consumer
	 * 	consumer to communicate with the main code
	 * @param str
	 * 	String to communicate
	 */
	public static void addProgress(Consumer<String> consumer, String str) {
		addProgress(consumer, str, false);
	}
	
	/**
	 * Write the wanted string to a Consumer to communicate with the main interface.
	 * @param consumer
	 * 	consumer to communicate with the main code
	 * @param str
	 * 	String to communicate
	 * @param addTime
	 * 	whether to specify the time along with the message or not
	 */
	public static void addProgress(Consumer<String> consumer, String str, boolean addTime) {
		if (consumer == null)
			return;
		if (addTime)
			str = Log.gct() + " -- " + str + System.lineSeparator();
		else
			str += System.lineSeparator();
		consumer.accept(str);
	}
}
