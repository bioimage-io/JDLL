package io.bioimage.modelrunner.system;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import org.junit.Test;

public class GpuCompatibilityTest {
    @Test
    public void windowsDriverVersionsMapToNvidiaVersions() {
        assertEquals("531.14", GpuCompatibility.mapWindowsWmiDriverVersionToNvidia("31.0.15.3114").get());
        assertEquals("537.42", GpuCompatibility.mapWindowsWmiDriverVersionToNvidia("31.0.15.3742").get());
        assertEquals("472.12", GpuCompatibility.mapWindowsWmiDriverVersionToNvidia("30.0.14.7212").get());
        assertEquals("581.80", GpuCompatibility.mapWindowsWmiDriverVersionToNvidia("32.0.15.8180\r\n").get());
        assertEquals("607.06", GpuCompatibility.mapWindowsWmiDriverVersionToNvidia("32.0.16.0706").get());
        assertEquals("502.84", GpuCompatibility.mapWindowsWmiDriverVersionToNvidia("31.0.15.284").get());
    }

    @Test
    public void incompleteOrInvalidDriverVersionsAreNotConverted() {
        for (String version : new String[] {"", "31.0.15", "31.0.15.31140", "NVIDIA 531.14", "31.0.15.error"}) {
            assertFalse(version, GpuCompatibility.mapWindowsWmiDriverVersionToNvidia(version).isPresent());
        }
    }
}
