package io.bioimage.modelrunner.tensor.shm;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;

public class SharedMemory {
    public static final int O_RDWR = 0x0002;
    public static final int O_CREAT = 0x0200;
    public static final int PROT_READ = 0x1;
    public static final int PROT_WRITE = 0x2;
    public static final int MAP_SHARED = 0x01;

    public static void main(String[] args) {
        CLibrary lib = CLibrary.INSTANCE;

        String sharedMemoryName = "/my_shared_memory";
        int size = 1024; // Size in bytes

        // Create or open a shared memory object
        int fd = lib.shm_open(sharedMemoryName, O_CREAT | O_RDWR, 0666);
        if (fd == -1) {
            throw new RuntimeException("shm_open failed");
        }

        // Set the size of the shared memory object
        if (lib.ftruncate(fd, size) == -1) {
            throw new RuntimeException("ftruncate failed");
        }

        // Map the shared memory object into the address space
        Pointer ptr = lib.mmap(Pointer.NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == Pointer.NULL) {
            throw new RuntimeException("mmap failed");
        }

        // Example: Write to the shared memory
        ptr.setString(0, "Hello from Java and JNA!");

        // Unmap the shared memory
        if (lib.munmap(ptr, size) == -1) {
            throw new RuntimeException("munmap failed");
        }

        // Close the file descriptor
        if (lib.close(fd) == -1) {
            throw new RuntimeException("close failed");
        }

        // Unlink the shared memory object
        if (lib.shm_unlink(sharedMemoryName) == -1) {
            throw new RuntimeException("shm_unlink failed");
        }
        System.out.println(false);
    }
}

