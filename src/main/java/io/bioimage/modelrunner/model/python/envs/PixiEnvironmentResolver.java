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
package io.bioimage.modelrunner.model.python.envs;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Objects;

import org.apposed.appose.util.Environments;

import io.bioimage.modelrunner.system.PlatformDetection;

/**
 * Common helpers for rendering Pixi environment templates and resolving
 * platform-specific resources shipped inside the JDLL jar.
 */
public final class PixiEnvironmentResolver {

	private static final String DEFAULT_SELECTED_ENVIRONMENT = "default";
	private static final String JDLL_CACHE_DIR_NAME = "jdll";

	private PixiEnvironmentResolver() {
		// Utility class.
	}

	/**
	 * Renders a classpath Pixi template using {@link String#format(Locale, String, Object...)}.
	 *
	 * @param environmentDirectoryName
	 * 	name of the directory under the Appose environment directory
	 * @param tomlResource
	 * 	classpath resource containing the Pixi template
	 * @param args
	 * 	template arguments
	 * @return resolved Pixi environment spec
	 */
	public static PixiEnvironmentSpec fromTemplate(String environmentDirectoryName,
			String tomlResource, Object... args) {
		String pixiTomlContent = String.format(Locale.ROOT, readClasspathResourceAsString(tomlResource), environmentDirectoryName, args);
		File environmentDirectory = new File(Environments.apposeEnvsDir(), environmentDirectoryName);
		return new PixiEnvironmentSpec(DEFAULT_SELECTED_ENVIRONMENT, pixiTomlContent,
				environmentDirectory, new ArrayList<String>());
	}

	/**
	 * Caches a classpath resource under the user JDLL cache directory.
	 *
	 * @param resourcePath
	 * 	classpath resource path
	 * @param cacheSubdirName
	 * 	subdirectory under the JDLL cache directory
	 * @return cached file
	 */
	public static File cacheClasspathResource(String resourcePath, String cacheSubdirName) {
		Objects.requireNonNull(resourcePath, "resourcePath");
		Objects.requireNonNull(cacheSubdirName, "cacheSubdirName");

		File cacheDir = new File(userCacheDir(JDLL_CACHE_DIR_NAME), cacheSubdirName);
		if (!cacheDir.isDirectory() && !cacheDir.mkdirs()) {
			throw new RuntimeException("Could not create cache directory: " + cacheDir.getAbsolutePath());
		}
		try (InputStream is = PixiEnvironmentResolver.class.getClassLoader().getResourceAsStream(resourcePath)) {
			if (is == null) {
				throw new RuntimeException("Required resource not found on classpath: " + resourcePath);
			}
			byte[] content = readAllBytesJava8(is);
			String fileName = resourcePath.substring(resourcePath.lastIndexOf('/') + 1);
			File contentCacheDir = new File(cacheDir, sha256Hex(content).substring(0, 12));
			if (!contentCacheDir.isDirectory() && !contentCacheDir.mkdirs()) {
				throw new RuntimeException("Could not create cache directory: " + contentCacheDir.getAbsolutePath());
			}
			File cachedFile = new File(contentCacheDir, fileName);
			if (cachedFile.isFile() && cachedFile.length() == content.length) {
				return cachedFile;
			}
			Files.write(cachedFile.toPath(), content);
			return cachedFile;
		} catch (IOException e) {
			throw new RuntimeException("Could not cache classpath resource: " + resourcePath, e);
		}
	}

	private static String sha256Hex(byte[] content) {
		try {
			MessageDigest digest = MessageDigest.getInstance("SHA-256");
			byte[] hash = digest.digest(content);
			StringBuilder hex = new StringBuilder(hash.length * 2);
			for (byte b : hash) {
				hex.append(String.format(Locale.ROOT, "%02x", b & 0xff));
			}
			return hex.toString();
		} catch (NoSuchAlgorithmException e) {
			throw new RuntimeException("SHA-256 is not available.", e);
		}
	}

	/**
	 * Reads a classpath resource as UTF-8 text.
	 *
	 * @param resourcePath
	 * 	classpath resource path
	 * @return resource contents
	 */
	public static String readClasspathResourceAsString(String resourcePath) {
		Objects.requireNonNull(resourcePath, "resourcePath");
		try (InputStream is = PixiEnvironmentResolver.class.getClassLoader().getResourceAsStream(resourcePath)) {
			if (is == null) {
				throw new RuntimeException("Required resource not found on classpath: " + resourcePath);
			}
			return new String(readAllBytesJava8(is), StandardCharsets.UTF_8);
		} catch (IOException e) {
			throw new RuntimeException("Failed to read classpath resource: " + resourcePath, e);
		}
	}

	/**
	 * Selects one resource path for the current OS and architecture.
	 *
	 * @param linuxX86
	 * 	resource for Linux x86_64
	 * @param linuxArm
	 * 	resource for Linux arm64/aarch64
	 * @param macX86
	 * 	resource for macOS x86_64/Rosetta
	 * @param macArm
	 * 	resource for macOS arm64/aarch64
	 * @param winX86
	 * 	resource for Windows x86_64
	 * @param description
	 * 	human-readable resource description for errors
	 * @return selected resource path
	 */
	public static String selectResourceByCurrentPlatform(String linuxX86, String linuxArm,
			String macX86, String macArm, String winX86, String description) {
		String arch = PlatformDetection.getArch();
		if (PlatformDetection.isLinux()) {
			if (PlatformDetection.ARCH_X86_64.equals(arch)) {
				return linuxX86;
			}
			if (PlatformDetection.ARCH_ARM64.equals(arch) || PlatformDetection.ARCH_AARCH64.equals(arch)) {
				return linuxArm;
			}
		} else if (PlatformDetection.isMacOS()) {
			if (PlatformDetection.ARCH_X86_64.equals(arch) || PlatformDetection.isUsingRosseta()) {
				return macX86;
			}
			if (PlatformDetection.ARCH_ARM64.equals(arch) || PlatformDetection.ARCH_AARCH64.equals(arch)) {
				return macArm;
			}
		} else if (PlatformDetection.isWindows() && PlatformDetection.ARCH_X86_64.equals(arch)) {
			return winX86;
		}
		throw new RuntimeException("Unsupported platform for " + description + ": "
				+ PlatformDetection.getOs() + "-" + arch);
	}

	/**
	 * @return current Pixi platform string.
	 */
	public static String currentPixiPlatform() {
		String arch = PlatformDetection.getArch();
		if (PlatformDetection.isLinux()) {
			if (PlatformDetection.ARCH_X86_64.equals(arch)) {
				return "linux-64";
			}
			if (PlatformDetection.ARCH_ARM64.equals(arch) || PlatformDetection.ARCH_AARCH64.equals(arch)) {
				return "linux-aarch64";
			}
		} else if (PlatformDetection.isMacOS()) {
			if ((PlatformDetection.ARCH_ARM64.equals(arch)
					|| PlatformDetection.ARCH_AARCH64.equals(arch))
					&& !PlatformDetection.isUsingRosseta()) {
				return "osx-arm64";
			}
			return "osx-64";
		} else if (PlatformDetection.isWindows() && PlatformDetection.ARCH_X86_64.equals(arch)) {
			return "win-64";
		}
		throw new RuntimeException("Unsupported Pixi platform: " + PlatformDetection.getOs() + "-" + arch);
	}

	/**
	 * Converts a local file path to a Pixi-friendly path string.
	 *
	 * @param file
	 * 	file to format
	 * @return absolute path using forward slashes
	 */
	public static String toPixiPath(File file) {
		return file.getAbsolutePath().replace('\\', '/');
	}

	/**
	 * Returns the per-user cache directory for the given application subfolder.
	 *
	 * @param appSubdir
	 * 	application cache subdirectory name
	 * @return cache directory
	 */
	public static File userCacheDir(String appSubdir) {
		String base;
		if (PlatformDetection.isWindows()) {
			String localAppData = System.getenv("LOCALAPPDATA");
			if (localAppData != null && !localAppData.trim().isEmpty()) {
				base = localAppData;
			} else {
				base = new File(System.getProperty("user.home"), "AppData" + File.separator + "Local").getAbsolutePath();
			}
		} else if (PlatformDetection.isMacOS()) {
			base = new File(System.getProperty("user.home"), "Library" + File.separator + "Caches").getAbsolutePath();
		} else {
			String xdgCache = System.getenv("XDG_CACHE_HOME");
			if (xdgCache != null && !xdgCache.trim().isEmpty()) {
				base = xdgCache;
			} else {
				base = new File(System.getProperty("user.home"), ".cache").getAbsolutePath();
			}
		}
		return new File(base, appSubdir);
	}

	private static byte[] readAllBytesJava8(InputStream is) throws IOException {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		byte[] buffer = new byte[8192];
		int len;
		while ((len = is.read(buffer)) != -1) {
			baos.write(buffer, 0, len);
		}
		return baos.toByteArray();
	}
}
