package io.bioimage.modelrunner.tensor;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

public interface POSIX extends Library {
    POSIX INSTANCE = Native.load("c", POSIX.class);

    int shm_open(String name, int flags, int mode);
    int ftruncate(int fd, long length);
    Pointer mmap(Pointer addr, long length, int prot, int flags, int fd, long offset);
    int munmap(Pointer addr, long length);
    int close(int fd);
    int shm_unlink(String name);

    // Constants are typically defined in sys/mman.h and fcntl.h
    int PROT_READ = 0x1;
    int PROT_WRITE = 0x2;
    int PROT_EXEC = 0x4;
    int MAP_SHARED = 0x01;
    int MAP_PRIVATE = 0x02;
    int O_RDONLY = 0x0000;
    int O_RDWR = 0x0002;
    int O_CREAT = 0x0040;
    int O_EXCL = 0x0080;
}
