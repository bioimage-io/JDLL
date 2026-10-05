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
package io.bioimage.modelrunner.gui.custom.training;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Stream;

/** Run-scoped mailbox. Each empty file is one idempotent full-validation request. */
public final class ValidationRequests implements AutoCloseable {
    private final Path directory = Files.createTempDirectory("jdll-validation-");
    private boolean ready;
    private boolean closed;

    public ValidationRequests() throws IOException {}

    public String getDirectory() { return directory.toString(); }

    public synchronized void accept(Map<String, Object> event) {
        if (!"full_validation".equals(event.get("type"))) return;
        if ("ready".equals(event.get("status"))) ready = Boolean.TRUE.equals(event.get("supported"));
        if ("closed".equals(event.get("status"))) ready = false;
    }

    public synchronized String request() throws IOException {
        if (!ready || closed) return null;
        String id = UUID.randomUUID().toString();
        Files.createFile(directory.resolve(id + ".request"));
        return id;
    }

    @Override
    public synchronized void close() throws IOException {
        if (closed) return;
        closed = true;
        ready = false;
        try (Stream<Path> files = Files.list(directory)) {
            for (Path file : (Iterable<Path>) files::iterator) Files.deleteIfExists(file);
        }
        Files.deleteIfExists(directory);
    }
}
