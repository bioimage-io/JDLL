package io.bioimage.modelrunner.model.python;

import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.TaskException;

/**
 * Helpers for invoking Appose from plugin classloader based applications.
 * <p>
 * This is an Icy/Groovy classloader compatibility workaround. Appose 0.11 uses
 * Groovy JSON for task messages, and Groovy's {@code FastStringUtils} discovers
 * {@code FastStringServiceFactory} through {@link java.util.ServiceLoader} with
 * the current thread context classloader. In production Icy, worker threads can
 * have a context classloader that does not see the shaded Groovy provider inside
 * {@code JDLLLibrary.jar}. The symptom is:
 * <pre>
 * java.lang.RuntimeException: Unable to load FastStringService
 *   at org.apache.groovy.json.internal.FastStringUtils.getService(...)
 * </pre>
 * The Appose task code itself is valid; only the context classloader used during
 * task message encoding is wrong. These wrappers temporarily switch the context
 * classloader to JDLL's classloader while creating and starting Appose tasks.
 * <p>
 * Removal path: if Icy starts plugin worker threads with a context classloader
 * that can see JDLL/Groovy service resources, or if Appose/Groovy no longer
 * relies on the thread context classloader for message encoding, replace
 * {@link #task(Service, String)} with {@code service.task(code)}, replace
 * {@link #waitFor(Task)} with {@code task.waitFor()}, and delete this class.
 */
public final class ApposeTaskUtils {

	@FunctionalInterface
	public interface ApposeCallable<T> {
		T call() throws InterruptedException, TaskException;
	}

	@FunctionalInterface
	public interface ApposeRunnable {
		void run() throws InterruptedException, TaskException;
	}

	private ApposeTaskUtils() {
		// Utility class.
	}

	public static <T> T withJdllContextClassLoader(final ApposeCallable<T> callable)
			throws InterruptedException, TaskException {
		Thread thread = Thread.currentThread();
		ClassLoader previous = thread.getContextClassLoader();
		ClassLoader jdllClassLoader = ApposeTaskUtils.class.getClassLoader();
		if (jdllClassLoader == null || jdllClassLoader == previous) {
			return callable.call();
		}
		thread.setContextClassLoader(jdllClassLoader);
		try {
			return callable.call();
		} finally {
			thread.setContextClassLoader(previous);
		}
	}

	public static void withJdllContextClassLoader(final ApposeRunnable runnable)
			throws InterruptedException, TaskException {
		withJdllContextClassLoader(() -> {
			runnable.run();
			return null;
		});
	}

	public static Task task(final Service service, final String code)
			throws InterruptedException, TaskException {
		return withJdllContextClassLoader(() -> service.task(code));
	}

	public static void waitFor(final Task task)
			throws InterruptedException, TaskException {
		withJdllContextClassLoader(() -> task.waitFor());
	}
}
