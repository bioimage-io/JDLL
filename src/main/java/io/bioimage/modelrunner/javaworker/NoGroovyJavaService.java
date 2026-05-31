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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;

/**
 * Minimal process-backed service for JDLL Java workers.
 * <p>
 * It preserves the Appose-like line-delimited JSON protocol used by the Java
 * engines, but it avoids Appose's Groovy-backed Java message serializer.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class NoGroovyJavaService implements AutoCloseable {

	private final Process process;
	private final BufferedWriter stdin;
	private final Map<String, NoGroovyTask> tasks = new ConcurrentHashMap<String, NoGroovyTask>();
	private volatile Consumer<String> debug = new Consumer<String>() {
		@Override
		public void accept(String text) {
			// Default is quiet.
		}
	};

	/**
	 * Starts a worker process.
	 * 
	 * @param directory the process working directory.
	 * @param environment environment entries to override.
	 * @param command the process command.
	 * @throws IOException if the process cannot be started.
	 */
	public NoGroovyJavaService(File directory, Map<String, String> environment, List<String> command) throws IOException {
		ProcessBuilder builder = new ProcessBuilder(command);
		if (directory != null)
			builder.directory(directory);
		if (environment != null) {
			for (Map.Entry<String, String> entry : environment.entrySet()) {
				if (entry.getKey() != null && entry.getValue() != null)
					builder.environment().put(entry.getKey(), entry.getValue());
			}
		}
		this.process = builder.start();
		this.stdin = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8));
		startReader(process.getInputStream(), true);
		startReader(process.getErrorStream(), false);
		startWatcher();
	}

	/**
	 * Registers a debug sink.
	 * 
	 * @param debug the debug line consumer.
	 */
	public void debug(Consumer<String> debug) {
		this.debug = debug == null ? new Consumer<String>() {
			@Override
			public void accept(String text) {
				// Quiet.
			}
		} : debug;
	}

	/**
	 * Starts a worker task.
	 * 
	 * @param script task name.
	 * @return the task.
	 * @throws IOException if the request cannot be written.
	 */
	public NoGroovyTask task(String script) throws IOException {
		return task(script, null);
	}

	/**
	 * Starts a worker task.
	 * 
	 * @param script task name.
	 * @param inputs task inputs.
	 * @return the task.
	 * @throws IOException if the request cannot be written.
	 */
	public NoGroovyTask task(String script, Map<String, Object> inputs) throws IOException {
		String id = UUID.randomUUID().toString();
		NoGroovyTask task = new NoGroovyTask(id);
		tasks.put(id, task);
		Map<String, Object> request = new HashMap<String, Object>();
		request.put("task", id);
		request.put("requestType", NoGroovyMessages.REQUEST_EXECUTE);
		request.put("script", script);
		request.put("inputs", inputs == null ? Collections.emptyMap() : inputs);
		send(request);
		return task;
	}

	/**
	 * Requests cancellation of a task.
	 * 
	 * @param task the task to cancel.
	 * @throws IOException if the request cannot be written.
	 */
	public void cancel(NoGroovyTask task) throws IOException {
		if (task == null)
			return;
		Map<String, Object> request = new HashMap<String, Object>();
		request.put("task", task.id);
		request.put("requestType", NoGroovyMessages.REQUEST_CANCEL);
		send(request);
	}

	private synchronized void send(Map<String, Object> request) throws IOException {
		stdin.write(NoGroovyMessages.encode(request));
		stdin.newLine();
		stdin.flush();
	}

	private void startReader(final InputStream stream, final boolean stdout) {
		Thread thread = new Thread(new Runnable() {
			@Override
			public void run() {
				readLoop(stream, stdout);
			}
		}, stdout ? "JDLL JavaWorker stdout" : "JDLL JavaWorker stderr");
		thread.setDaemon(true);
		thread.start();
	}

	private void readLoop(InputStream stream, boolean stdout) {
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
			String line;
			while ((line = reader.readLine()) != null) {
				if (stdout && handleResponse(line))
					continue;
				debug.accept(line);
			}
		} catch (IOException e) {
			debug.accept(NoGroovyMessages.stackTrace(e));
		}
	}

	private boolean handleResponse(String line) {
		Map<String, Object> response;
		try {
			response = NoGroovyMessages.decode(line);
		} catch (RuntimeException e) {
			return false;
		}
		if (response == null)
			return false;
		Object idObj = response.get("task");
		Object typeObj = response.get("responseType");
		if (!(idObj instanceof String) || !(typeObj instanceof String))
			return false;
		NoGroovyTask task = tasks.get(idObj);
		if (task == null)
			return false;
		String type = (String) typeObj;
		if (NoGroovyMessages.RESPONSE_LAUNCH.equals(type)) {
			task.running();
		} else if (NoGroovyMessages.RESPONSE_COMPLETION.equals(type)) {
			task.complete(response);
			tasks.remove(idObj);
		} else if (NoGroovyMessages.RESPONSE_FAILURE.equals(type)) {
			Object error = response.get("error");
			task.fail(error == null ? null : error.toString());
			tasks.remove(idObj);
		} else if (NoGroovyMessages.RESPONSE_CANCELATION.equals(type)) {
			task.cancel();
			tasks.remove(idObj);
		} else if (NoGroovyMessages.RESPONSE_UPDATE.equals(type)) {
			debug.accept(line);
		}
		return true;
	}

	private void startWatcher() {
		Thread thread = new Thread(new Runnable() {
			@Override
			public void run() {
				try {
					int exit = process.waitFor();
					crashPending("Worker process terminated with exit code " + exit + ".");
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
					crashPending(NoGroovyMessages.stackTrace(e));
				}
			}
		}, "JDLL JavaWorker watcher");
		thread.setDaemon(true);
		thread.start();
	}

	private void crashPending(String error) {
		for (NoGroovyTask task : tasks.values())
			task.crash(error);
		tasks.clear();
	}

	/**
	 * Closes the worker process.
	 */
	@Override
	public void close() {
		try {
			stdin.close();
		} catch (IOException e) {
			debug.accept(NoGroovyMessages.stackTrace(e));
		}
		process.destroy();
	}
}
