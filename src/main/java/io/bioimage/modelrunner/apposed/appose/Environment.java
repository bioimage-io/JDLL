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

package io.bioimage.modelrunner.apposed.appose;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import io.bioimage.modelrunner.system.PlatformDetection;

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
		String[] allArgs;
		if (exeFile == null) throw new IllegalArgumentException("No executables found amongst candidates: " + exes);

		/*if ((exeFile.getName().equals("python") || exeFile.getName().equals("python.exe"))
				&& PlatformDetection.isMacOS() && !PlatformDetection.getArch().equals(PlatformDetection.ARCH_ARM64)
				&& PlatformDetection.isUsingRosseta()) {
			allArgs = new String[args.length + 3];
			System.arraycopy(args, 0, allArgs, 3, args.length);
			allArgs[0] = "arch";
			allArgs[1] = "-arm64";
			allArgs[2] = exeFile.getCanonicalPath();
		} else {
			allArgs = new String[args.length + 1];
			System.arraycopy(args, 0, allArgs, 1, args.length);
			allArgs[0] = exeFile.getCanonicalPath();
		}*/
		allArgs = new String[args.length + 1];
		System.arraycopy(args, 0, allArgs, 1, args.length);
		allArgs[0] = exeFile.getCanonicalPath();

		return new Service(new File(base()), allArgs);
	}
}
