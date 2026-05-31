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

import java.util.Map;
import java.util.concurrent.CountDownLatch;

/**
 * Minimal task object used by {@link NoGroovyJavaService}.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class NoGroovyTask {

	public enum TaskStatus {
		PENDING,
		RUNNING,
		COMPLETE,
		FAILED,
		CANCELED,
		CRASHED
	}

	public volatile TaskStatus status = TaskStatus.PENDING;
	public volatile String error;
	public volatile Map<String, Object> outputs;

	final String id;
	private final CountDownLatch done = new CountDownLatch(1);

	NoGroovyTask(String id) {
		this.id = id;
	}

	void running() {
		this.status = TaskStatus.RUNNING;
	}

	@SuppressWarnings("unchecked")
	void complete(Map<String, Object> response) {
		Object outs = response.get("outputs");
		if (outs instanceof Map<?, ?>)
			this.outputs = (Map<String, Object>) outs;
		this.status = TaskStatus.COMPLETE;
		done.countDown();
	}

	void fail(String error) {
		this.error = error;
		this.status = TaskStatus.FAILED;
		done.countDown();
	}

	void cancel() {
		this.status = TaskStatus.CANCELED;
		done.countDown();
	}

	void crash(String error) {
		this.error = error;
		this.status = TaskStatus.CRASHED;
		done.countDown();
	}

	/**
	 * Blocks until the worker sends a terminal response or the worker process
	 * crashes.
	 * 
	 * @throws InterruptedException if interrupted while waiting.
	 */
	public void waitFor() throws InterruptedException {
		done.await();
	}
}
