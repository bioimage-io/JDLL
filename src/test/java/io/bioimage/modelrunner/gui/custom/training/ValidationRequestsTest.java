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

import static org.junit.Assert.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;

public class ValidationRequestsTest {
    @Test
    public void tokensAreRunScopedAndReadyGatedAndCleanedUp() throws Exception {
        Path directory;
        try (ValidationRequests requests = new ValidationRequests()) {
            directory = Paths.get(requests.getDirectory());
            assertNull(requests.request());
            requests.accept(event("ready", true));
            String first = requests.request();
            assertTrue(Files.exists(directory.resolve(first + ".request")));
            requests.accept(event("started", true));
            String second = requests.request();
            assertNotEquals(first, second);
            requests.accept(event("completed", true));
            assertTrue(Files.exists(directory.resolve(second + ".request")));
            requests.accept(event("failed", true));
            assertNotNull(requests.request());
            requests.accept(event("closed", true));
            assertNull(requests.request());
        }
        assertFalse(Files.exists(directory));
    }

    private static Map<String, Object> event(String status, boolean supported) {
        Map<String, Object> value = new HashMap<>();
        value.put("type", "full_validation");
        value.put("status", status);
        value.put("supported", supported);
        return value;
    }
}
