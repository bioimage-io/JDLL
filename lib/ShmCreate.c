#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>

// Function to create a shared memory segment, modified to accept a long for size
int create_shared_memory(const char *name, long size) {
    int fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        perror("shm_open");
        return -1;
    }
    int size = get_shared_memory_size(fd);
    if (size > 0) {
        return fd;
    }

    if (ftruncate(fd, size) == -1) {
        perror("ftruncate");
        close(fd);
        return -1;
    }

    return fd;
}

// Function to unlink a shared memory segment
void unlink_shared_memory(const char *name) {
    if (shm_unlink(name) == -1) {
        perror("shm_unlink");
    }
}

// Function to get the size of a shared memory segment given its file descriptor
long get_shared_memory_size(int fd) {
    struct stat shm_stat;
    if (fstat(fd, &shm_stat) == -1) {
        perror("fstat");
        return -1;
    }
    return (long)shm_stat.st_size;
}

int main() {
    const char *shm_name = "/myshm";
    size_t shm_size = 1024;

    // Create shared memory
    int shm_fd = create_shared_memory(shm_name, shm_size);
    if (shm_fd < 0) {
        exit(EXIT_FAILURE);
    }

    // Perform operations with shared memory here
    // ...

    // Close the shared memory file descriptor
    close(shm_fd);

    // Unlink shared memory
    unlink_shared_memory(shm_name);

    return 0;
}

