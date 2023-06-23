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
/**
 * 
 */
package io.bioimage.modelrunner.engine;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;

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
	 * Auxiliary URLClassLoader that only contains the classes that
	 * can be read from the URLs provided in the constructor
	 */
	private ClassLoader helper;

    /**
     * Create a child-first/parent-last URLClassLoader
     * @param urls
     * 	the URLS to be included in the classpath of the new classloader
     * @param parent
     * 	parent classloader
     */
		public ParentLastURLClassLoader(URL[] urls, ClassLoader parent) {
			super(urls, null);
			helper = parent;
		}

	    @Override
	    /**
	     * {@inheritDoc}
	     * 
	     * For the parent-last classloader, first, if the class is already loaded, just recovers
	     * it, and if it has to laod it, it tries first to load the class from the JARs,
	     * and if it is not possible, it falls back to the parent classloader
	     */
		protected synchronized Class<?> loadClass(String name, boolean resolve)
				throws ClassNotFoundException {
			// First, check if the class has already been loaded
			Class<?> c = findLoadedClass(name);
			if (c == null) {
				try {
					// checking local
					c = findClass(name);
				} catch (ClassNotFoundException e) {
					c = loadClassFromParent(name, resolve);
				} catch(SecurityException e){
					c = loadClassFromParent(name, resolve);
				}
			}
			if (resolve)
				resolveClass(c);
			return c;
		}

	    /**
	     * Check the super classloader and the parent
	     * This call to loadClass may eventually call findClass
	     * again, in case the parent doesn't find anything.
	     * @param name
	     * 	the name of the class
	     * @param resolve
	     * 	If true then resolve the class
	     * @return the loaded class
	     * @throws ClassNotFoundException if the class is not found
	     */
		private Class<?> loadClassFromParent(String name, boolean resolve) throws ClassNotFoundException {
			Class<?> c;
			try {
				c = super.loadClass(name, resolve);
			} catch (ClassNotFoundException e) {
				c = loadClassFromSystem(name);
			} catch (SecurityException e){
				c = loadClassFromSystem(name);
			}
			return c;
		}

		/**
		 * Try to load a class from the parent classloader
		 * @param name
		 * 	name of the class
		 * @return the loaded class
		 * @throws ClassNotFoundException if the class is not found
		 */
		private Class<?> loadClassFromSystem(String name) throws ClassNotFoundException{
			Class<?> c = null;
			if (helper != null) {
				// checking system: jvm classes, endorsed, cmd classpath,
				// etc.
				c = helper.loadClass(name);
			}
			return c;
		}

		@Override
		/**
		 * {@inheritDoc}
		 * 
		 * The child classloader resource has a bigger priority
		 */
		public URL getResource(String name) {
			URL url = findResource(name);
			if (url == null)
				url = super.getResource(name);

			if (url == null && helper != null)
				url = helper.getResource(name);

			return url;
		}
	    
	    /**
	     * {@inheritDoc}
	     * 
	     * This custom classloader checks first if the resources can be found in any of the
	     * urls provided in the constructor
	     */
	    @Override
		public Enumeration<URL> getResources(String name) throws IOException {
			/**
			 * Similar to super, but local resources are enumerated before parent
			 * resources
			 */
			Enumeration<URL> systemUrls = null;
			if (helper != null) {
				systemUrls = helper.getResources(name);
			}
			Enumeration<URL> localUrls = findResources(name);
			Enumeration<URL> parentUrls = null;
			if (getParent() != null) {
				parentUrls = getParent().getResources(name);
			}
			final List<URL> urls = new ArrayList<URL>();
			if (localUrls != null) {
				while (localUrls.hasMoreElements()) {
					URL local = localUrls.nextElement();
					urls.add(local);
				}
			} else if (parentUrls != null && localUrls == null) {
				while (parentUrls.hasMoreElements()) {
					urls.add(parentUrls.nextElement());
				}
			} else if (systemUrls != null && localUrls == null && parentUrls == null) {
				while (systemUrls.hasMoreElements()) {
					urls.add(systemUrls.nextElement());
				}
			}
			return new Enumeration<URL>() {
				Iterator<URL> iter = urls.iterator();

				public boolean hasMoreElements() {
					return iter.hasNext();
				}

				public URL nextElement() {
					return iter.next();
				}
			};
		}

	    /**
	     * REturn the resource specified by the name.
	     * The child resource has a bigger priority
	     * @param name
	     * 	name of the resource
	     * @return input stream containing teh resource
	     */
		public InputStream getResourceAsStream(String name) {
			URL url = getResource(name);
			try {
				return url != null ? url.openStream() : null;
			} catch (IOException e) {
			}
			return null;
		}
	}