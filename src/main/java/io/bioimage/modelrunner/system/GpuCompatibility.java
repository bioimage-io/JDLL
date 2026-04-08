/*-
 * #%L
 * Library to call models of the family of SAM (Segment Anything Model) from Java
 * %%
 * Copyright (C) 2024 SAMJ developers.
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
package io.bioimage.modelrunner.system;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Cross-platform, dependency-free checks for NVIDIA GPU + driver compatibility with CUDA Toolkit versions.
 *
 * Notes:
 * - Linux: uses /sys and /proc first (fast, no extra tools), then optional command fallbacks.
 * - Windows: uses built-in PowerShell CIM queries (present by default) and checks for nvcuda.dll.
 * - macOS: CUDA is not supported on modern macOS; methods will reflect that (GPU may exist historically, but CUDA stack is not).
 *
 * Table of minimum driver versions comes from NVIDIA CUDA Toolkit Release Notes (Table 3).
 * 
 * @author Carlos Garcia
 */
public final class GpuCompatibility {

    // ---------- CUDA Toolkit -> minimum driver table (Linux + Windows) ----------

    /**
     * Minimum driver versions per CUDA Toolkit (GA) from NVIDIA release notes Table 3.
     * We store major.minor only and use the GA row minima.
     *
     * Source: NVIDIA CUDA Toolkit Release Notes PDF (Table 3). :contentReference[oaicite:1]{index=1}
     */
    private static final Map<String, DriverMin> MIN_DRIVER_BY_CUDA;
    static {
        Map<String, DriverMin> m = new LinkedHashMap<>();

        // CUDA 12.x (GA minima)
        m.put("12.0", new DriverMin("525.60.13", "527.41"));
        m.put("12.1", new DriverMin("530.30.02", "531.14"));
        m.put("12.2", new DriverMin("535.54.03", "536.25"));
        m.put("12.3", new DriverMin("545.23.06", "545.84"));
        m.put("12.4", new DriverMin("550.54.14", "551.61"));
        m.put("12.5", new DriverMin("555.42.02", "555.85"));
        m.put("12.6", new DriverMin("560.28.03", "560.76"));
        // Table shows 12.8 GA and 12.9; 12.7 not listed in that excerpt
        m.put("12.8", new DriverMin("570.26", "570.65"));
        m.put("12.9", new DriverMin("575.51.03", "576.02"));

        // (Optional) A bit of 11.x that is still common
        m.put("11.8", new DriverMin("520.61.05", "520.06"));
        m.put("11.7", new DriverMin("515.43.04", "516.01"));
        m.put("11.6", new DriverMin("510.39.01", "511.23"));
        m.put("11.5", new DriverMin("495.29.05", "496.04"));
        m.put("11.4", new DriverMin("470.42.01", "471.11"));
        m.put("11.3", new DriverMin("465.19.01", "465.89"));
        m.put("11.2", new DriverMin("460.27.03", "460.82"));
        m.put("11.1", new DriverMin("455.23", "456.38"));
        m.put("11.0", new DriverMin("450.51.05", "451.48"));

        MIN_DRIVER_BY_CUDA = Collections.unmodifiableMap(m);
    }

    // Prevent instantiation
    private GpuCompatibility() {}

    // ---------- Public API ----------

    /**
	 * @return True if an NVIDIA GPU is present (hardware detection). 
	 */
    public static boolean hasCudaCapableNvidiaGpu() {
        Os os = Os.detect();
        switch (os.family) {
            case LINUX:
                // Best universal Linux method: PCI vendor ID 0x10de in sysfs.
                if (linuxHasPciVendor("0x10de")) return true;
                // Fallback: try lspci if present.
                String out = runCmdBestEffort(1500, "sh", "-lc", "command -v lspci >/dev/null 2>&1 && lspci");
                return containsIgnoreCase(out, "nvidia corporation") || containsIgnoreCase(out, "vga compatible controller: nvidia");
            case WINDOWS:
                // Use CIM/WMI via PowerShell (built-in).
                String names = runPowerShell(2500,
                        "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name");
                return containsIgnoreCase(names, "nvidia");
            case MAC:
                // system_profiler is built-in. (Apple Silicon will not have NVIDIA.)
                String sp = runCmdBestEffort(3500, "sh", "-lc", "system_profiler SPDisplaysDataType 2>/dev/null");
                return containsIgnoreCase(sp, "nvidia");
            default:
                return false;
        }
    }

    /**
     * @return True if NVIDIA driver appears installed and loaded enough to support CUDA.
     * - Linux: checks /proc/driver/nvidia/version OR loaded modules OR /dev/nvidiactl.
     * - Windows: checks nvcuda.dll existence OR NVIDIA display driver presence via CIM.
     * - macOS: returns false (CUDA driver stack not supported).
     */
    public static boolean areNvidiaDriversInstalled() {
        Os os = Os.detect();
        switch (os.family) {
            case LINUX:
                if (Files.exists(Paths.get("/proc/driver/nvidia/version"))) return true;
                if (Files.exists(Paths.get("/dev/nvidiactl"))) return true;
                // Check loaded modules from /proc/modules (no external tools).
                String modules = readFileBestEffort("/proc/modules");
                if (!modules.isEmpty()) {
                    boolean hasNvidia = containsLineStartingWith(modules, "nvidia ");
                    boolean hasUvm = containsLineStartingWith(modules, "nvidia_uvm ");
                    boolean hasDrm = containsLineStartingWith(modules, "nvidia_drm ");
                    // User asked specifically for these 3; require all three.
                    return hasNvidia && hasUvm && hasDrm;
                }
                // Fallback to lsmod if present.
                String lsmod = runCmdBestEffort(1500, "sh", "-lc", "command -v lsmod >/dev/null 2>&1 && lsmod");
                return containsLineWithExactToken(lsmod, "nvidia")
                        && containsLineWithExactToken(lsmod, "nvidia_uvm")
                        && containsLineWithExactToken(lsmod, "nvidia_drm");
            case WINDOWS:
                // nvcuda.dll is the CUDA Driver API library (provided by the NVIDIA driver).
                if (windowsHasFileInSystem32("nvcuda.dll")) return true;
                // Also check for NVIDIA video controller existence (driver installed).
                String names = runPowerShell(2500,
                        "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name");
                return containsIgnoreCase(names, "nvidia");
            case MAC:
                return false;
            default:
                return false;
        }
    }

    /**
     * @return Returns the NVIDIA driver version in NVIDIA-style "535.129.03" when possible.
     * - Linux: parses /proc/driver/nvidia/version
     * - Windows: converts WMI DriverVersion like "31.0.15.3114" -> "531.14" (best-effort)
     * - macOS: empty (not supported)
     */
    public static Optional<String> getNvidiaDriverVersion() {
        Os os = Os.detect();
        switch (os.family) {
            case LINUX:
                String v = readFileBestEffort("/proc/driver/nvidia/version");
                // Example: "NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.129.03  ..."
                Matcher m = Pattern.compile("\\b(\\d{3}\\.\\d{1,3}\\.\\d{1,3})\\b").matcher(v);
                if (m.find()) return Optional.of(m.group(1));
                return Optional.empty();
            case WINDOWS:
                // WMI DriverVersion often looks like: 31.0.15.3114 (maps to 531.14)
                String drv = runPowerShell(3000,
                        "Get-CimInstance Win32_VideoController | " +
                        "Where-Object {$_.Name -match 'NVIDIA'} | " +
                        "Select-Object -First 1 -ExpandProperty DriverVersion");
                drv = drv.trim();
                if (drv.isEmpty()) return Optional.empty();
                Optional<String> mapped = mapWindowsWmiDriverVersionToNvidia(drv);
                return mapped.isPresent() ? mapped : Optional.of(drv);
            case MAC:
                return Optional.empty();
            default:
                return Optional.empty();
        }
    }

    /** @return True if the CUDA driver API library is present (Linux: libcuda.so.1, Windows: nvcuda.dll). */
    public static boolean hasLibCudaDriverApi() {
        Os os = Os.detect();
        switch (os.family) {
            case LINUX:
                // Fast: check common locations first
                if (linuxHasAnyExistingFile(
                        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                        "/usr/lib64/libcuda.so.1",
                        "/usr/lib/libcuda.so.1",
                        "/lib/x86_64-linux-gnu/libcuda.so.1",
                        "/lib64/libcuda.so.1",
                        "/lib/libcuda.so.1"
                )) return true;

                // If ldconfig exists, use it (but don't depend on it).
                String out = runCmdBestEffort(1500, "sh", "-lc", "command -v ldconfig >/dev/null 2>&1 && ldconfig -p");
                return containsIgnoreCase(out, "libcuda.so.1") || containsIgnoreCase(out, "libcuda.so");
            case WINDOWS:
                return windowsHasFileInSystem32("nvcuda.dll");
            case MAC:
                return false;
            default:
                return false;
        }
    }

    /**
     * @return whether the current driver is compatible with the requested CUDA Toolkit version (e.g., "12.1").
     * Uses a built-in table of minimum required drivers for Linux and Windows (NVIDIA release notes).
     */
    public static boolean isDriverCompatibleWithCuda(String cudaVersion) {
        Os os = Os.detect();
        if (os.family == OsFamily.MAC) return false;

        Version requestedCuda = Version.parseMajorMinor(cudaVersion).orElse(null);
        if (requestedCuda == null) return false;

        DriverMin req = MIN_DRIVER_BY_CUDA.get(requestedCuda.major + "." + requestedCuda.minor);
        if (req == null) {
            // Not in our table. Conservative choice: if CUDA is 12.x, require >= 525 (minor version compatibility floor),
            // but for specific toolkits, the per-version minimum is safer. Return false if unknown.
            return false;
        }

        Optional<String> drvStrOpt = getNvidiaDriverVersion();
        if (!drvStrOpt.isPresent()) return false;

        Version driver = Version.parseFlexible(drvStrOpt.get()).orElse(null);
        if (driver == null) return false;

        Version min = (os.family == OsFamily.WINDOWS) ? req.windowsMin : req.linuxMin;
        if (min == null) return false;

        return driver.compareTo(min) >= 0;
    }

    /**
     * @return whether the wanted CUDA version can be installed in an environment or not
     * Full "can I install CUDA X.Y in an environment and run it here?" gate:
     * - NVIDIA GPU present
     * - drivers installed
     * - CUDA Driver API present (libcuda / nvcuda.dll)
     * - driver version meets the minimum for the requested toolkit version
     *
     * Note: This checks host capability only. It does not install anything.
     */
    public static boolean canInstallCudaInEnv(String cudaVersion) {
        Os os = Os.detect();
        if (os.family == OsFamily.MAC) return false; // practical reality

        return hasCudaCapableNvidiaGpu()
                && areNvidiaDriversInstalled()
                && hasLibCudaDriverApi()
                && isDriverCompatibleWithCuda(cudaVersion);
    }

    /** @return a list of all CUDA versions from our table 
     * that are compatible with this machine's current driver.
     */
    public static List<String> getCompatibleCudaVersions() {
        Os os = Os.detect();
        if (os.family == OsFamily.MAC) return Collections.emptyList();

        Optional<String> drvStrOpt = getNvidiaDriverVersion();
        if (!drvStrOpt.isPresent()) return Collections.emptyList();

        Version driver = Version.parseFlexible(drvStrOpt.get()).orElse(null);
        if (driver == null) return Collections.emptyList();

        List<String> result = new ArrayList<>();
        for (Map.Entry<String, DriverMin> e : MIN_DRIVER_BY_CUDA.entrySet()) {
            DriverMin req = e.getValue();
            Version min = (os.family == OsFamily.WINDOWS) ? req.windowsMin : req.linuxMin;
            if (min != null && driver.compareTo(min) >= 0) {
                result.add(e.getKey());
            }
        }

        // Sort by version numeric ascending
        result.sort((a, b) -> {
            Version va = Version.parseMajorMinor(a).orElse(new Version(0,0,0));
            Version vb = Version.parseMajorMinor(b).orElse(new Version(0,0,0));
            return va.compareTo(vb);
        });
        return result;
    }

    /**
     * Returns the first CUDA version from the provided list that can be installed
     * in the current environment.
     *
     * <p>The versions are checked in order, and the first one for which
     * {@code GpuCompatibility.canInstallCudaInEnv(String)} returns {@code true}
     * is returned.</p>
     *
     * @param compatibleCudas
     *     the list of compatible CUDA versions to evaluate, in priority order
     * @return the first installable CUDA version, or {@code null} if none of the
     *     provided versions can be installed
     */
    public static String pickCudaVersion(List<String> compatibleCudas) {
        for (String cv : compatibleCudas) {
            if (GpuCompatibility.canInstallCudaInEnv(cv)) {
                return cv;
            }
        }
        return null;
    }

    // ---------- Helpers & internals ----------

    private enum OsFamily { WINDOWS, LINUX, MAC, OTHER }

    private static final class Os {
        final OsFamily family;
        final String arch;
        private Os(OsFamily family, String arch) { this.family = family; this.arch = arch; }

        static Os detect() {
            String name = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
            String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);

            if (name.contains("win")) return new Os(OsFamily.WINDOWS, arch);
            if (name.contains("linux")) return new Os(OsFamily.LINUX, arch);
            if (name.contains("mac") || name.contains("darwin")) return new Os(OsFamily.MAC, arch);
            return new Os(OsFamily.OTHER, arch);
        }
    }

    /** Simple version struct that supports flexible parsing like 535.129.03, 531.14, 575.51.03 */
    private static final class Version implements Comparable<Version> {
        final int major, minor, patch;
        Version(int major, int minor, int patch) { this.major = major; this.minor = minor; this.patch = patch; }

        static Optional<Version> parseMajorMinor(String s) {
            if (s == null) return Optional.empty();
            Matcher m = Pattern.compile("^\\s*(\\d+)\\.(\\d+)\\s*$").matcher(s.trim());
            if (!m.find()) return Optional.empty();
            return Optional.of(new Version(Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2)), 0));
        }

        static Optional<Version> parseFlexible(String s) {
            if (s == null) return Optional.empty();
            String t = s.trim();
            // Accept "535.129.03" or "531.14" or "570.26"
            Matcher m = Pattern.compile("^(\\d+)\\.(\\d+)(?:\\.(\\d+))?$").matcher(t);
            if (!m.find()) return Optional.empty();
            int a = Integer.parseInt(m.group(1));
            int b = Integer.parseInt(m.group(2));
            int c = (m.group(3) != null) ? Integer.parseInt(m.group(3)) : 0;
            return Optional.of(new Version(a, b, c));
        }

        @Override public int compareTo(Version o) {
            if (major != o.major) return Integer.compare(major, o.major);
            if (minor != o.minor) return Integer.compare(minor, o.minor);
            return Integer.compare(patch, o.patch);
        }

        @Override public String toString() { return major + "." + minor + "." + patch; }
    }

    private static final class DriverMin {
        final Version linuxMin;
        final Version windowsMin;
        DriverMin(String linuxMin, String windowsMin) {
            this.linuxMin = Version.parseFlexible(linuxMin).orElse(null);
            this.windowsMin = Version.parseFlexible(windowsMin).orElse(null);
        }
    }

    private static boolean linuxHasPciVendor(String vendorHexLower) {
        // /sys/bus/pci/devices/*/vendor contains e.g. 0x10de for NVIDIA
        Path pci = Paths.get("/sys/bus/pci/devices");
        if (!Files.isDirectory(pci)) return false;
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(pci)) {
            for (Path dev : ds) {
                Path vendor = dev.resolve("vendor");
                if (Files.isRegularFile(vendor)) {
                    String v = new String(Files.readAllBytes(vendor), StandardCharsets.US_ASCII).trim().toLowerCase(Locale.ROOT);
                    if (v.equals(vendorHexLower)) return true;
                }
            }
        } catch (IOException ignored) {}
        return false;
    }

    private static boolean linuxHasAnyExistingFile(String... paths) {
        for (String p : paths) {
            if (Files.exists(Paths.get(p))) return true;
        }
        return false;
    }

    private static boolean windowsHasFileInSystem32(String filename) {
        String winDir = System.getenv("WINDIR");
        if (winDir == null || winDir.trim().isEmpty()) return false;
        Path p = Paths.get(winDir, "System32", filename);
        return Files.exists(p);
    }

    private static Optional<String> mapWindowsWmiDriverVersionToNvidia(String wmi) {
        // Typical: 31.0.15.3114 -> NVIDIA 531.14  (take last 5 digits: 53114 -> 531.14)
        Matcher m = Pattern.compile(".*\\.(\\d{5})\\s*$").matcher(wmi.trim());
        if (!m.find()) return Optional.empty();
        String five = m.group(1);
        String maj = five.substring(0, 3);
        String min = five.substring(3);
        return Optional.of(maj + "." + min);
    }

    private static String readFileBestEffort(String path) {
        try {
            Path p = Paths.get(path);
            if (!Files.exists(p)) return "";
            byte[] b = Files.readAllBytes(p);
            // keep it small
            String s = new String(b, StandardCharsets.UTF_8);
            return s.length() > 64_000 ? s.substring(0, 64_000) : s;
        } catch (IOException e) {
            return "";
        }
    }

    private static boolean containsIgnoreCase(String haystack, String needle) {
        if (haystack == null || needle == null) return false;
        return haystack.toLowerCase(Locale.ROOT).contains(needle.toLowerCase(Locale.ROOT));
    }

    private static boolean containsLineStartingWith(String text, String prefix) {
        for (String line : text.split("\\R")) {
            if (line.startsWith(prefix)) return true;
        }
        return false;
    }

    private static boolean containsLineWithExactToken(String text, String token) {
        Pattern p = Pattern.compile("(^|\\s)" + Pattern.quote(token) + "(\\s|$)");
        for (String line : text.split("\\R")) {
            if (p.matcher(line).find()) return true;
        }
        return false;
    }

    private static String runPowerShell(long timeoutMs, String command) {
        // PowerShell is built into Windows; on non-Windows this will just fail fast and return "".
        return runCmdBestEffort(timeoutMs,
                "powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-Command", command);
    }

    private static String runCmdBestEffort(long timeoutMs, String... cmd) {
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);

        try {
            Process p = pb.start();
            boolean done = p.waitFor(timeoutMs, TimeUnit.MILLISECONDS);
            if (!done) {
                p.destroyForcibly();
                return "";
            }
            try (InputStream is = p.getInputStream()) {
                byte[] out = readUpTo(is, 64_000);
                return new String(out, StandardCharsets.UTF_8);
            }
        } catch (Exception e) {
            return "";
        }
    }

    private static byte[] readUpTo(InputStream is, int maxBytes) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[4096];
        int total = 0;
        int r;
        while ((r = is.read(buf)) != -1) {
            if (total + r > maxBytes) {
                bos.write(buf, 0, maxBytes - total);
                break;
            }
            bos.write(buf, 0, r);
            total += r;
            if (total >= maxBytes) break;
        }
        return bos.toByteArray();
    }
}