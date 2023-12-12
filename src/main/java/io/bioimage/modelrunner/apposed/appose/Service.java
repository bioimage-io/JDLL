/*-
 * #%L
 * Appose: multi-language interprocess cooperation with shared memory.
 * %%
 * Copyright (C) 2023 Appose developers.
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */

package io.bioimage.modelrunner.apposed.appose;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;

import io.bioimage.modelrunner.apposed.appose.TaskEvent;

/**
 * An Appose *service* provides access to a linked Appose *worker* running in a
 * different process. Using the service, programs create Appose {@link Task}s
 * that run asynchronously in the worker process, which notifies the service of
 * updates via communication over pipes (stdin and stdout).
 */
public class Service implements AutoCloseable {

	private static int serviceCount = 0;

	private final File cwd;
	private final String[] args;
	private final Map<String, Task> tasks = new ConcurrentHashMap<>();
	private final int serviceID;

	private Process process;
	private PrintWriter stdin;
	private Thread stdoutThread;
	private Thread stderrThread;
	private Thread monitorThread;

	private Consumer<String> debugListener;

	public Service(File cwd, String... args) {
		this.cwd = cwd;
		this.args = args.clone();
		serviceID = serviceCount++;
	}

	/**
	 * Registers a callback function to receive messages
	 * describing current service/worker activity.
	 *
	 * @param debugListener A function that accepts a single string argument.
	 */
	public void debug(Consumer<String> debugListener) {
		this.debugListener = debugListener;
	}

	/**
	 * Launches the worker process associated with this service.
	 *
	 * @return This service object, for chaining method calls (typically with {@link #task}).
	 * @throws IOException If the process fails to execute; see {@link ProcessBuilder#start()}.
	 */
	public Service start() throws IOException {
		if (process != null) {
			// Already started.
			return this;
		}

		String prefix = "Appose-Service-" + serviceID;
		ProcessBuilder pb = new ProcessBuilder(args).directory(cwd);
		process = pb.start();
		stdin = new PrintWriter(process.getOutputStream());
		stdoutThread = new Thread(this::stdoutLoop, prefix + "-Stdout");
		stderrThread = new Thread(this::stderrLoop, prefix + "-Stderr");
		monitorThread = new Thread(this::monitorLoop, prefix + "-Monitor");
		stderrThread.start();
		stdoutThread.start();
		monitorThread.start();
		return this;
	}

	/**
	 * Creates a new task, passing the given script to the worker for execution.
	 *
	 * @param script The script for the worker to execute in its environment.
	 * @return The newly created {@link Task} object tracking the execution.
	 * @throws IOException If something goes wrong communicating with the worker.
	 */
	public Task task(String script) throws IOException {
		return task(script, null);
	}

	/**
	 * Creates a new task, passing the given script to the worker for execution.
	 *
	 * @param script The script for the worker to execute in its environment.
	 * @param inputs Optional list of key/value pairs to feed into the script as inputs.
	 * @return The newly created {@link Task} object tracking the execution.
	 * @throws IOException If something goes wrong communicating with the worker.
	 */
	public Task task(String script, Map<String, Object> inputs) throws IOException {
		start();
		return new Task(script, inputs);
	}

	/** Closes the worker process's input stream, in order to shut it down. */
	@Override
	public void close() {
		stdin.close();
	}

	/** Input loop processing lines from the worker stdout stream. */
	private void stdoutLoop() {
		BufferedReader stdout = new BufferedReader(new InputStreamReader(process.getInputStream()));
		while (true) {
			String line;
			try {
				line = stdout.readLine();
			}
			catch (IOException exc) {
				// Something went wrong reading the line. Panic!
				debugService(Types.stackTrace(exc));
				break;
			}

			if (line == null) {
				debugService("<worker stdout closed>");
				return;
			}
			try {
				Map<String, Object> response = Types.decode(line);
				debugService(line); // Echo the line to the debug listener.
				Object uuid = response.get("task");
				if (uuid == null) {
					debugService("Invalid service message:" + line);
					continue;
				}
				Task task = tasks.get(uuid.toString());
				if (task == null) {
					debugService("No such task: " + uuid);
					continue;
				}
				task.handle(response);
			}
			catch (Exception exc) {
				// Something went wrong decoding the line of JSON.
				// Skip it and keep going, but log it first.
				debugService(String.format("<INVALID> %s", line));
			}
		}
	}

	/** Input loop processing lines from the worker stderr stream. */
	private void stderrLoop() {
		BufferedReader stderr = new BufferedReader(new InputStreamReader(process.getErrorStream()));
		try {
			while (true) {
				String line = stderr.readLine();
				if (line == null) {
					debugService("<worker stderr closed>");
					return;
				}
				debugWorker(line);
			}
		}
		catch (IOException exc) {
			debugWorker(Types.stackTrace(exc));
		}
	}

	private void monitorLoop() {
		// Wait until the worker process terminates.
		while (process.isAlive()) {
			try {
				Thread.sleep(50);
			}
			catch (InterruptedException exc) {
				debugService(Types.stackTrace(exc));
			}
		}

		// Do some sanity checks.
		int exitCode = process.exitValue();
		if (exitCode != 0) debugService("<worker process terminated with exit code " + exitCode + ">");
		int taskCount = tasks.size();
		if (taskCount > 0) debugService("<worker process terminated with " + taskCount + " pending tasks>");

		// Notify any remaining tasks about the process crash.
		tasks.values().forEach(Task::crash);
		tasks.clear();
	}

	private void debugService(String message) { debug("SERVICE", message); }
	private void debugWorker(String message) { debug("WORKER", message); }

	/**
	 * Passes a message to the listener registered
	 * via the {@link #debug(Consumer)} method.
	 */
	private void debug(String prefix, String message) {
		if (debugListener == null) return;
		debugListener.accept("[" + prefix + "-" + serviceID + "] " + message);
	}

	public enum TaskStatus {
		INITIAL, QUEUED, RUNNING, COMPLETE, CANCELED, FAILED, CRASHED;

		/**
		 * @return true iff status is {@link #COMPLETE}, {@link #CANCELED}, {@link #FAILED}, or {@link #CRASHED}.
		 */
		public boolean isFinished() {
			return this == COMPLETE || this == CANCELED || this == FAILED || this == CRASHED;
		}
	}

	public enum RequestType {
		EXECUTE, CANCEL
	}

	public enum ResponseType {
		LAUNCH, UPDATE, COMPLETION, CANCELATION, FAILURE, CRASH
	}

	/**
	 * An Appose *task* is an asynchronous operation performed by its associated
	 * Appose {@link Service}. It is analogous to a {@code Future}.
	 */
	public class Task {

		public final String uuid = UUID.randomUUID().toString();
		public final String script;
		private final Map<String, Object> mInputs = new HashMap<>();
		private final Map<String, Object> mOutputs = new HashMap<>();
		public final Map<String, Object> inputs = Collections.unmodifiableMap(mInputs);
		public final Map<String, Object> outputs = Collections.unmodifiableMap(mOutputs);

		public TaskStatus status = TaskStatus.INITIAL;
		public String message;
		public long current;
		public long maximum = 1;
		public String error;

		private final List<Consumer<TaskEvent>> listeners = new ArrayList<>();

		public Task(String script, Map<String, Object> inputs) {
			this.script = script;
			if (inputs != null) mInputs.putAll(inputs);
			tasks.put(uuid, this);
		}

		public synchronized Task start() {
			if (status != TaskStatus.INITIAL) throw new IllegalStateException();
			status = TaskStatus.QUEUED;

			Map<String, Object> args = new HashMap<>();
			args.put("script", script);
			args.put("inputs", inputs);
			request(RequestType.EXECUTE, args);

			return this;
		}

		/**
		 * Registers a listener to be notified of updates to the task.
		 *
		 * @param listener Function to invoke in response to task status updates.
		 */
		public synchronized void listen(Consumer<TaskEvent> listener) {
			if (status != TaskStatus.INITIAL) {
				throw new IllegalStateException("Task is not in the INITIAL state");
			}
			listeners.add(listener);
		}

		public synchronized void waitFor() throws InterruptedException {
			if (status == TaskStatus.INITIAL) start();
			if (status != TaskStatus.QUEUED && status != TaskStatus.RUNNING) return;
			wait();
		}

		/** Sends a task cancelation request to the worker process. */
		public void cancel() {
			request(RequestType.CANCEL, null);
		}

		/** Sends a request to the worker process. */
		private void request(RequestType requestType, Map<String, Object> args) {
			Map<String, Object> request = new HashMap<>();
			request.put("task", uuid);
			request.put("requestType", requestType.toString());
			if (args != null) request.putAll(args);
			String encoded = Types.encode(request);

			stdin.println(encoded);
			// NB: Flush is necessary to ensure worker receives the data!
			stdin.flush();
			debugService(encoded);
		}

		@SuppressWarnings("hiding")
		private void handle(Map<String, Object> response) {
			String maybeResponseType = (String) response.get("responseType");
			if (maybeResponseType == null) {
				debugService("Message type not specified");
				return;
			}
			ResponseType responseType = ResponseType.valueOf(maybeResponseType);

			switch (responseType) {
				case LAUNCH:
					status = TaskStatus.RUNNING;
					break;
				case UPDATE:
					message = (String) response.get("message");
					Number current = (Number) response.get("current");
					Number maximum = (Number) response.get("maximum");
					if (current != null) this.current = current.longValue();
					if (maximum != null) this.maximum = maximum.longValue();
					break;
				case COMPLETION:
					tasks.remove(uuid);
					status = TaskStatus.COMPLETE;
					@SuppressWarnings({ "rawtypes", "unchecked" })
					Map<String, Object> outputs = (Map) response.get("outputs");
					if (outputs != null) mOutputs.putAll(outputs);
					break;
				case CANCELATION:
					tasks.remove(uuid);
					status = TaskStatus.CANCELED;
					break;
				case FAILURE:
					tasks.remove(uuid);
					status = TaskStatus.FAILED;
					Object error = response.get("error");
					this.error = error == null ? null : error.toString();
					break;
				default:
					debugService("Invalid service message type: " + responseType);
					return;
			}

			TaskEvent event = new TaskEvent(this, responseType);
			listeners.forEach(l -> l.accept(event));

			if (status.isFinished()) {
				synchronized (this) {
					notifyAll();
				}
			}
		}

		private void crash() {
			TaskEvent event = new TaskEvent(this, ResponseType.CRASH);
			status = TaskStatus.CRASHED;
			listeners.forEach(l -> l.accept(event));
			synchronized (this) {
				notifyAll();
			}
		}

		@Override
		public String toString() {
			return String.format("uuid=%s, status=%s, message=%s, current=%d, maximum=%d, error=%s",
				uuid, status, message, current, maximum, error);
		}
	}
}
