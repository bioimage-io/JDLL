/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the BioImage.io nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
/**
 * 
 */
package io.bioimage.modelrunner.engine;

import java.net.URL;
import java.net.URLClassLoader;

/**
 * Custom classloader that acts as an URL classloader that tries to load first the classes in
 * the URL classloader and then in the parent.
 * This was necessary because the Javacpp Loader looked for the JARs added to the engine classloader
 * to look for the directories where the native libraries where (inside the JAR file).
 * Due to the special classloading protocol in Fiji, the engine classloader never added 
 * the JARs to the classpath, thus the native libraries were never found. With a child-first/parent-last
 * the classes from the JARs are always loaded on the child classloader (engineclassloader) making its 
 * classpath include the JARs that contain the native libraries.
 * See  org.bytedeco.javacpp#findResources at line 894 to find the conflicting part, where
 * if the classloader was not parent-last, the needed resource was not found
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class ParentLastURLClassLoader extends URLClassLoader {

    /**
     * Create a child-first/parent-last URLClassLoader
     * @param urls
     * 	the URLS to be included in the classpath of the new classloader
     * @param parent
     * 	parent classloader
     */
    public ParentLastURLClassLoader(URL[] urls, ClassLoader parent) {
        super(urls, parent);
    }

    @Override
    /**
     * {@inheritDoc}
     * 
     * For the parent-last classloader, first, if the class is already loaded, just recovers
     * it, and if it has to laod it, it tries first to load the class from the JARs,
     * and if it is not possible, it falls back to the parent classloader
     */
    protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
        // First, check if the class is already loaded
        Class<?> loadedClass = findLoadedClass(name);
        if (loadedClass != null) {
            return loadedClass;
        }

        try {
            // Try to load the class from the URLs of this classloader
            Class<?> localClass = findClass(name);
            if (resolve) {
                resolveClass(localClass);
            }
            return localClass;
        } catch (ClassNotFoundException e) {
            // Class not found in this classloader, delegate to parent classloader
            return super.loadClass(name, resolve);
        }
    }
}