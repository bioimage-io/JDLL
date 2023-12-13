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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
//TODO remove once appose project is released with the needed changes
public interface Environment {

	default String base() { return "."; }
	default boolean useSystemPath() { return false; }

	/**
	 * Creates a Python script service.
	 * <p>
	 * This is a <b>high level</b> way to create a service, enabling execution of
	 * Python scripts asynchronously on its linked process running a
	 * {@code python_worker}.
	 * </p>
	 * 
	 * @return The newly created service.
	 * @throws IOException If something goes wrong starting the worker process.
	 */
	default Service python() throws IOException {
		List<String> pythonExes = Arrays.asList(
			"python", "python.exe",
			"bin/python", "bin/python.exe"
		);
		return service(pythonExes, "-c",
			"import appose.python_worker; appose.python_worker.main()");
	}

	default Service java(String mainClass, String... jvmArgs)
		throws IOException
	{
		return java(mainClass, Collections.emptyList(), jvmArgs);
	}

	default Service java(String mainClass, List<String> classPath,
		String... jvmArgs) throws IOException
	{
		// Collect classpath elements into a set, to avoid duplicate entries.
		Set<String> cp = new LinkedHashSet<>();

		// Ensure that the classpath includes Appose and its dependencies.
		// NB: This list must match Appose's dependencies in pom.xml!
		List<Class<?>> apposeDeps = Arrays.asList(//
			com.sun.jna.Pointer.class, // ------------------> com.sun.jna:jna
			com.sun.jna.platform.linux.LibRT.class, // -----> com.sun.jna:jna-platform
			com.sun.jna.platform.win32.Kernel32.class // ---> com.sun.jna:jna-platform
		);
		for (Class<?> depClass : apposeDeps) {
			File location = FilePaths.location(depClass);
			if (location != null) cp.add(location.getCanonicalPath());
		}

		// Append any explicitly requested classpath elements.
		cp.addAll(classPath);

		// Build up the service arguments.
		List<String> args = new ArrayList<>();
		args.add("-cp");
		args.add(String.join(File.pathSeparator, cp));
		args.addAll(Arrays.asList(jvmArgs));
		args.add(mainClass);

		// Create the service.
		List<String> javaExes = Arrays.asList(
			"java", "java.exe",
			"bin/java", "bin/java.exe",
			"jre/bin/java", "jre/bin/java.exe"
		);
		return service(javaExes, args.toArray(new String[0]));
	}

	/**
	 * Creates a service with the given command line arguments.
	 * <p>
	 * This is a <b>low level</b> way to create a service. It assumes the
	 * specified executable conforms to the {@link Appose Appose worker process
	 * contract}, meaning it accepts requests on stdin and produces responses on
	 * stdout, both formatted according to Appose's assumptions.
	 * </p>
	 *
	 * @param exes List of executables to try for launching the worker process.
	 * @param args Command line arguments to pass to the worker process
	 *          (e.g. <code>{"-v", "--enable-everything"}</code>.
	 * @return The newly created service.
	 * @see #python() To create a service for Python script execution.
	 * @throws IOException If something goes wrong starting the worker process.
	 */
	default Service service(List<String> exes, String... args) throws IOException {
		if (args.length == 0) throw new IllegalArgumentException("No executable given");

		List<String> dirs = useSystemPath() //
			? Arrays.asList(System.getenv("PATH").split(File.pathSeparator)) //
			: Arrays.asList(base());

		File exeFile = FilePaths.findExe(dirs, exes);
		if (exeFile == null) throw new IllegalArgumentException("No executables found amongst candidates: " + exes);

		String[] allArgs = new String[args.length + 1];
		System.arraycopy(args, 0, allArgs, 1, args.length);
		allArgs[0] = exeFile.getCanonicalPath();

		return new Service(new File(base()), allArgs);
	}
}
