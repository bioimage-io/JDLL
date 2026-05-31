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
package io.bioimage.modelrunner.tensor.shm;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;


public interface CLibrary extends Library {
    CLibrary INSTANCE = Native.load("c", CLibrary.class);

    /**
     * Opens a POSIX shared memory object.
     *
     * @param name the name.
     * @param oflag the open flags.
     * @param mode the mode.
     * @return the native status code.
     */
    int shm_open(String name, int oflag, int mode);
    /**
     * Truncates a file descriptor to the requested length.
     *
     * @param fd the file descriptor.
     * @param length the length.
     * @return the native status code.
     */
    int ftruncate(int fd, int length);
    /**
     * Maps memory for a file descriptor.
     *
     * @param addr the address.
     * @param length the length.
     * @param prot the memory protection flags.
     * @param flags the flags.
     * @param fd the file descriptor.
     * @param offset the offset.
     * @return the mapped memory pointer.
     */
    Pointer mmap(Pointer addr, int length, int prot, int flags, int fd, int offset);
    /**
     * Unmaps a memory region.
     *
     * @param addr the address.
     * @param length the length.
     * @return the native status code.
     */
    int munmap(Pointer addr, int length);
    /**
     * Closes the native or shared memory resource.
     *
     * @param fd the file descriptor.
     * @return the native status code.
     */
    int close(int fd);
    /**
     * Unlinks a POSIX shared memory object.
     *
     * @param name the name.
     * @return the native status code.
     */
    int shm_unlink(String name);
    
    int SEEK_SET = 0;
    int SEEK_CUR = 1;
    int SEEK_END = 2;
    /**
     * Repositions the file offset.
     *
     * @param fd the file descriptor.
     * @param offset the offset.
     * @param whence the whence.
     * @return the native result value.
     */
    long lseek(int fd, long offset, int whence);
}
