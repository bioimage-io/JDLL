/**
 * 
 */
package org.bioimageanalysis.icy.deeplearning.utils;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class EngineLoader  extends ClassLoader{

	/**
	 * Child ClassLoader of the System ClassLoader that 
	 * adds the JARs needed to the classes 
	 */
	private ClassLoader engineClassloader;
	/**
	 * Child ClassLoader of the System ClassLoader that 
	 * adds the JARs needed to the classes 
	 */
	private ClassLoader icyClassloader;
	/**
	 * Path to the folder containing all the jars 
	 * needed to load the corresponding engine
	 */
	private String enginePath;
	/**
	 * Instance of the class from the wanted Deep Learning engine
	 * that is used to call all the needed methods to execute a model
	 */
	private DeepLearningInterface engineInstance;
	/**
	 * Name of the interface all the engines have to implement
	 */
	private static final String interfaceName = "org.bioimageanalysis.icy.deeplearning.utils.DeepLearningInterface";
	/**
	 * HashMap containing all the already loaded ClassLoader. This variables
	 * avoids reloading classes that have already been loaded. Reloading already
	 * existing ClassLaoders can be a problem if the loaded class opens an
	 * external native library because this library will not be freed until
	 * the Classes from the ClassLoader are Garbage Collected
	 */
	private static HashMap<String, ClassLoader> loadedEngines = new HashMap<String, ClassLoader>();
	
	/**
	 * Create a ClassLaoder that contains the classes of the parent
	 * ClassLoader given as an input and the classes found in the 
	 * String path given
	 * 
	 * @param classloader
	 * 	parent ClassLoader of the wanted ClassLoader
	 * @param enginePath
	 * 	String path where the new JARs containing the wanted classes
	 * 	should be located
	 */
	public EngineLoader(ClassLoader classloader, String enginePath) 
	{
		super();
		this.icyClassloader = classloader;
		this.enginePath = enginePath;
		loadClasses();
	    Thread.currentThread().setContextClassLoader(this.engineClassloader);
	    setEngineInstance();
	}
	
	/**
	 * Create an EngineLoader which creates an URLClassLoader
	 * with all the jars in the provided String path
	 */
	public EngineLoader(String enginePath) 
	{
		this(Thread.currentThread().getContextClassLoader(), enginePath);
	}
	
	/**
	 * Load the needed JAR files into a child ClassLoader of the 
	 * ContextClassLoader
	 */
	private void loadClasses() {
		// If the ClassLoader was already created, use it
		if (loadedEngines.get(enginePath) != null) {
			this.engineClassloader = loadedEngines.get(enginePath);
			return;
		}
		try {
		    /*URL url = new File(this.enginePath).toURI().toURL();         
		    URL[] urls = new URL[]{url};*/
			URL[] urls = new URL[new File(this.enginePath).listFiles().length];
			int c = 0;
			for (File ff : new File(this.enginePath).listFiles()) {
				urls[c ++] = ff.toURI().toURL();
			}
		    this.engineClassloader = new URLClassLoader(urls, this.icyClassloader);
		} 
		catch (MalformedURLException e) 
		{
		    // TODO refine exception
		}
		loadedEngines.put(this.enginePath, this.engineClassloader);
	}
	
	/**
	 * Returns the ClassLoader of the corresponding Deep Learning framework (engine)
	 * @param enginePath
	 * 	the path to the directory where all the JARs needed to load the corresponding
	 * 	Deep Learning framework (engine) are stored
	 * @return the ClassLoader corresponding to the wanted Deep Learning version
	 */
	public static EngineLoader createEngine(String enginePath)
	{
		return new EngineLoader(enginePath);
	}
	
	/**
	 * Find the wanted interface {@link DeepLearningInterface} from the entries
	 * of a JAR file. REturns null if the interface is not in the entries
	 * @param entries
	 * 	entries of a JAR executable file
	 * @return the wanted class implementing the interface or null if it is not there
	 * @throws ClassNotFoundException 
	 * @throws IllegalAccessException 
	 * @throws InstantiationException 
	 */
	private static DeepLearningInterface getEngineClassFromEntries(Enumeration<? extends ZipEntry> entries, 
										ClassLoader engineClassloader) 
										throws ClassNotFoundException, 
										InstantiationException, 
										IllegalAccessException
	{
		while (entries.hasMoreElements()) {
            ZipEntry entry = (ZipEntry) entries.nextElement();
            String file = entry.getName();
            if (file.endsWith(".class") && !file.contains("$") && !file.contains("-")) {
            	String className = getClassNameInJAR(file);
            	Class<?> c = engineClassloader.loadClass(className);
            	Class[] intf = c.getInterfaces();
            	for (int j=0; j<intf.length; j++) {
					if (intf[j].getName().equals(interfaceName)){
						// Assume that DeepLearningInterface has no arguments for the constructor
						return (DeepLearningInterface) c.newInstance();
					}
				}
            	// REmove references
            	intf = null;
            	c = null;
            }
        }
		return null;
	}
	
	/**
	 * Return the name of the class as seen by the ClassLoader from
	 * the name of the file entry in the JAR file. Basically removes the
	 * .class suffix and substitutes "/" by ".".
	 * 
	 * @param entryName
	 * 	String containing the name of the file compressed inside the JAR file
	 * @return the Class name as seen by the ClassLoader
	 */
	public static String getClassNameInJAR(String entryName)
	{
		String className = entryName.substring(0, entryName.indexOf("."));
		className = className.replace("/", ".");
		return className;
	}
	
	private void setEngineInstance()
	{
		// Load all the classes in the engine folder and select the wanted interface
		ZipFile jarFile;
		try {
			for (File ff : new File(this.enginePath).listFiles()) {
				jarFile = new ZipFile(ff);
				Enumeration<? extends ZipEntry> entries = jarFile.entries();
				this.engineInstance = getEngineClassFromEntries(entries, engineClassloader);
            	if (this.engineInstance != null) {
					jarFile.close();
					return;
				}
				jarFile.close();
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InstantiationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * Return the engine instance from where to call the corresponging engine
	 * @return engine instance
	 */
	public DeepLearningInterface getEngineInstance()
	{
		return this.engineInstance;
	}
	
	/**
	 * Close the created ClassLoader
	 * @throws IOException if closing any file opened 
	 * 	by this class loader resulted in an IOException. Any 
	 * 	such exceptions are caught internally. If only one is caught,
	 * 	then it is re-thrown. If more than one exception is caught, then
	 * 	the second and following exceptions are added as suppressed exceptions of 
	 * the first one caught, which is then re-thrown.
	 */
	// TODO is it necessary??
	public void close() {
		engineInstance.closeModel();
	    Thread.currentThread().setContextClassLoader(this.icyClassloader);
	    engineInstance = null;
	    engineClassloader = null;
	    loadedEngines = new HashMap<String, ClassLoader>();
	    System.out.print("Closed classloader");
	}
}
