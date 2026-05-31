package io.bioimage.modelrunner.model.python;

import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.apposed.appose.TaskException;

import io.bioimage.modelrunner.javaworker.NoGroovyMessages;

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
 * The Appose task code itself is valid; only Appose's Java-side Groovy JSON
 * encoder is fragile in Icy. For task launch/cancel, this class writes the same
 * line-delimited JSON protocol using JDLL's Gson encoder, then lets Appose read
 * and handle worker responses normally.
 * <p>
 * Removal path: if Icy starts plugin worker threads with a context classloader
 * that can see JDLL/Groovy service resources, or if Appose/Groovy no longer
 * relies on the thread context classloader for message encoding, replace
 * {@link #task(Service, String)} with {@code service.task(code)}, replace
 * {@link #waitFor(Task)} with {@code task.waitFor()}, replace
 * {@link #cancel(Task)} with {@code task.cancel()}, and delete this class.
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

	private static final Field SERVICE_STDIN_FIELD = field(Service.class, "stdin");

	private static final ClassLoader APPOSE_CONTEXT_CLASSLOADER = createApposeContextClassLoader();

	private static Field field(final Class<?> type, final String name) {
		try {
			Field field = type.getDeclaredField(name);
			field.setAccessible(true);
			return field;
		} catch (ReflectiveOperationException e) {
			throw new ExceptionInInitializerError(e);
		}
	}

	private static ClassLoader createApposeContextClassLoader() {
		List<ClassLoader> delegates = new ArrayList<>();
		addClassLoader(delegates, classLoaderOf("org.apache.groovy.json.FastStringServiceFactory"));
		addClassLoader(delegates, classLoaderOf("org.apache.groovy.json.DefaultFastStringServiceFactory"));
		addClassLoader(delegates, classLoaderOf("groovy.json.JsonGenerator"));
		addClassLoader(delegates, ApposeTaskUtils.class.getClassLoader());
		addClassLoader(delegates, Service.class.getClassLoader());
		addClassLoader(delegates, Thread.currentThread().getContextClassLoader());
		if (delegates.isEmpty()) {
			return null;
		}
		if (delegates.size() == 1) {
			return delegates.get(0);
		}
		return new ApposeContextClassLoader(delegates);
	}

	private static ClassLoader classLoaderOf(final String className) {
		List<ClassLoader> searchLoaders = new ArrayList<>();
		addClassLoader(searchLoaders, ApposeTaskUtils.class.getClassLoader());
		addClassLoader(searchLoaders, Service.class.getClassLoader());
		addClassLoader(searchLoaders, Thread.currentThread().getContextClassLoader());
		for (ClassLoader searchLoader : searchLoaders) {
			try {
				return Class.forName(className, false, searchLoader).getClassLoader();
			} catch (ClassNotFoundException e) {
				// Try the next visible loader.
			}
		}
		try {
			return Class.forName(className).getClassLoader();
		} catch (ClassNotFoundException e) {
			return null;
		}
	}

	private static void addClassLoader(final List<ClassLoader> classLoaders, final ClassLoader classLoader) {
		if (classLoader == null || classLoaders.contains(classLoader)) {
			return;
		}
		classLoaders.add(classLoader);
	}

	public static <T> T withJdllContextClassLoader(final ApposeCallable<T> callable)
			throws InterruptedException, TaskException {
		Thread thread = Thread.currentThread();
		ClassLoader previous = thread.getContextClassLoader();
		ClassLoader apposeClassLoader = APPOSE_CONTEXT_CLASSLOADER;
		if (apposeClassLoader == null || apposeClassLoader == previous) {
			return callable.call();
		}
		thread.setContextClassLoader(apposeClassLoader);
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
		startNoGroovyIfNeeded(task);
		withJdllContextClassLoader(() -> task.waitFor());
	}

	public static void cancel(final Task task) {
		try {
			sendNoGroovy(task, NoGroovyMessages.REQUEST_CANCEL, null);
		} catch (TaskException e) {
			throw new IllegalStateException("Could not cancel Appose task", e);
		}
	}

	private static void startNoGroovyIfNeeded(final Task task) throws TaskException {
		synchronized (task) {
			if (task.status != TaskStatus.INITIAL) {
				return;
			}
			task.status = TaskStatus.QUEUED;
		}

		Map<String, Object> payload = new HashMap<String, Object>();
		payload.put("script", task.script);
		payload.put("inputs", task.inputs == null ? Collections.emptyMap() : task.inputs);
		if (task.queue != null) {
			payload.put("queue", task.queue);
		}
		try {
			sendNoGroovy(task, NoGroovyMessages.REQUEST_EXECUTE, payload);
		} catch (TaskException e) {
			synchronized (task) {
				task.status = TaskStatus.INITIAL;
			}
			throw e;
		}
	}

	private static void sendNoGroovy(final Task task, final String requestType, final Map<String, Object> payload)
			throws TaskException {
		Map<String, Object> request = new HashMap<String, Object>();
		request.put("task", task.uuid);
		request.put("requestType", requestType);
		if (payload != null) {
			request.putAll(payload);
		}
		try {
			PrintWriter stdin = (PrintWriter) SERVICE_STDIN_FIELD.get(taskService(task));
			stdin.println(NoGroovyMessages.encode(request));
			stdin.flush();
		} catch (ReflectiveOperationException | RuntimeException e) {
			throw new TaskException("Could not send Appose task request without Groovy", task);
		}
	}

	private static Service taskService(final Task task) throws ReflectiveOperationException {
		Field serviceField = task.getClass().getDeclaredField("this$0");
		serviceField.setAccessible(true);
		return (Service) serviceField.get(task);
	}

	private static final class ApposeContextClassLoader extends ClassLoader {

		private final List<ClassLoader> delegates;

		ApposeContextClassLoader(final List<ClassLoader> delegates) {
			super(null);
			this.delegates = Collections.unmodifiableList(new ArrayList<>(delegates));
		}

		@Override
		protected Class<?> loadClass(final String name, final boolean resolve) throws ClassNotFoundException {
			ClassNotFoundException lastFailure = null;
			for (ClassLoader delegate : delegates) {
				try {
					return Class.forName(name, false, delegate);
				} catch (ClassNotFoundException e) {
					lastFailure = e;
				}
			}
			if (lastFailure != null) {
				throw lastFailure;
			}
			throw new ClassNotFoundException(name);
		}

		@Override
		public URL getResource(final String name) {
			for (ClassLoader delegate : delegates) {
				URL resource = delegate.getResource(name);
				if (resource != null) {
					return resource;
				}
			}
			return null;
		}

		@Override
		public Enumeration<URL> getResources(final String name) throws IOException {
			LinkedHashSet<URL> resources = new LinkedHashSet<>();
			for (ClassLoader delegate : delegates) {
				Enumeration<URL> delegateResources = delegate.getResources(name);
				while (delegateResources.hasMoreElements()) {
					resources.add(delegateResources.nextElement());
				}
			}
			return Collections.enumeration(resources);
		}
	}
}
