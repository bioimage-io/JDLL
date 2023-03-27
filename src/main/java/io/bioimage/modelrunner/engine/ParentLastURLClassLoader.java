package io.bioimage.modelrunner.engine;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;


public class ParentLastURLClassLoader extends URLClassLoader {
    private final List<String> hiddenClasses = new ArrayList<>();

    public ParentLastURLClassLoader(URL[] urls, ClassLoader parent) {
        super(urls, parent);
    }

    public void hideClass(String className) {
        hiddenClasses.add(className);
    }

    public void revealClass(String className) {
        hiddenClasses.remove(className);
    }

    @Override
    protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
        // First, check if the class is already loaded
        Class<?> loadedClass = findLoadedClass(name);
        if (loadedClass != null) {
            return loadedClass;
        }

        // Check if the class should be hidden
        if (hiddenClasses.contains(name)) {
            throw new ClassNotFoundException(name);
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