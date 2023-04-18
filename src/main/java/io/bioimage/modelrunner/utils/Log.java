package io.bioimage.modelrunner.utils;

import java.text.SimpleDateFormat;
import java.util.Date;

public class Log {
	/**
	 * Format of the time
	 */
	public static SimpleDateFormat format = new SimpleDateFormat("HH:mm:ss");

	/**
	 * REturns the time of the day in "HH:mm:ss" format with military time.
	 * Example 16:20:59
	 * @return the time
	 */
	public static String getCurrentTime() {
		return format.format(new Date());
		
	}
}
