package io.bioimage.modelrunner;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

public interface CLibrary extends Library {
    CLibrary INSTANCE = Native.load("c", CLibrary.class);

    int shm_open(String name, int oflag, int mode);
    int ftruncate(int fd, int length);
    Pointer mmap(Pointer addr, int length, int prot, int flags, int fd, int offset);
    int munmap(Pointer addr, int length);
    int close(int fd);
    int shm_unlink(String name);
}

