/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
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

    int shm_open(String name, int oflag, int mode);
    int ftruncate(int fd, int length);
    Pointer mmap(Pointer addr, int length, int prot, int flags, int fd, int offset);
    int munmap(Pointer addr, int length);
    int close(int fd);
    int shm_unlink(String name);
    
    int SEEK_SET = 0;
    int SEEK_CUR = 1;
    int SEEK_END = 2;
    long lseek(int fd, long offset, int whence);
    
    
    // TODO new
    int IPC_STAT = 1;

    // shmid_ds structure definition
    class shmid_ds extends com.sun.jna.Structure {
        public int shm_perm__pad1;
        public int shm_perm__pad2;
        public short shm_perm_mode;
        public short shm_perm__pad3;
        public int shm_perm_uid;
        public int shm_perm_gid;
        public long shm_perm__pad4;
        public long shm_segsz;
        public long shm_atime;
        public long shm_dtime;
        public long shm_ctime;
        public long shm_cpid;
        public long shm_lpid;
        public long shm_nattch;
        public long shm_unused1;
        public long shm_unused2;
    }

    int shmctl(int shmid, int cmd, shmid_ds buf);
}
