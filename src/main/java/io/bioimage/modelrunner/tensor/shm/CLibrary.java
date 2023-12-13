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

import java.util.Arrays;
import java.util.List;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.NativeLong;


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

    /*
    class Stat extends Structure {
        public long st_dev;
        public long st_ino;
        public long st_nlink;
        public int st_mode;
        public int st_uid;
        public int st_gid;
        public int __pad0;
        public long st_rdev;
        public long st_size;

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("st_dev", "st_ino", "st_nlink", "st_mode", "st_uid", "st_gid", "__pad0", "st_rdev", "st_size"); // add other fields as needed
        }
    }
    */
}
