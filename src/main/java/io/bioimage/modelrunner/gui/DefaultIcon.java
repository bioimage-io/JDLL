package io.bioimage.modelrunner.gui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URL;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.SwingUtilities;

public class DefaultIcon {

	protected static String DIJ_ICON_PATH;

    private static final Map<Dimension, CompletableFuture<ImageIcon>> PENDING_ICONS = new ConcurrentHashMap<>();
    private static Map<Dimension, ImageIcon> ICONS_CACHE = new ConcurrentHashMap<>();
    private static final Map<URL, CompletableFuture<BufferedImage>> PENDING = new ConcurrentHashMap<>();
    private static Map<URL, BufferedImage> CACHE = new ConcurrentHashMap<>();
    private static ExecutorService scaleExecutor = Executors.newFixedThreadPool(2);
    private static BufferedImage MASTER_IMAGE;
    private static String MASTER_PATH;
    
    protected static BufferedImage getImmediateLoadingSquareLogo() {
        BufferedImage bi = new BufferedImage(50, 50, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = bi.createGraphics();
        g.setColor(Color.GRAY);
        g.drawString("Loading...", 5, 25);
        g.dispose();
        return bi;
    }
    
    protected static BufferedImage getSoftwareLogo() {
        try {
        	if (DIJ_ICON_PATH == null)
        		throw new IOException();
        	if (MASTER_IMAGE != null && DIJ_ICON_PATH.equals(MASTER_PATH))
        		return MASTER_IMAGE;
            URL defaultIconUrl = DefaultIcon.class.getClassLoader().getResource(DIJ_ICON_PATH);
            if (defaultIconUrl == null) {
                throw new IOException();
            }
            BufferedImage cached = CACHE.get(defaultIconUrl);
            if (cached != null) {
                return cached;
            }
            
            if (scaleExecutor.isShutdown())
            	scaleExecutor = Executors.newFixedThreadPool(2);
            
            // Check if already being processed
            CompletableFuture<BufferedImage> pending = PENDING.get(defaultIconUrl);
            if (pending == null) {
                // Start new scaling operation
                pending = CompletableFuture.supplyAsync(() -> {
                    try {
						MASTER_IMAGE = ImageIO.read(defaultIconUrl);
						CACHE.put(defaultIconUrl, MASTER_IMAGE);
	                    MASTER_PATH = DIJ_ICON_PATH;
					} catch (IOException e) {
						e.printStackTrace();
						MASTER_IMAGE = getImmediateLoadingSquareLogo();
					}
                    PENDING.remove(defaultIconUrl);
                    return MASTER_IMAGE;
                }, scaleExecutor);
                PENDING.put(defaultIconUrl, pending);
            }
            return getImmediateLoadingSquareLogo();
        } catch (Exception ex) {
        	ex.printStackTrace();
        	return getImmediateLoadingSquareLogo();
        }
    }
    
    public static void drawImOrLogo(URL imURL, URL logoUrl, LogoPanel panel) {
        BufferedImage img = CACHE.get(imURL);
        if (img != null) {
        	SwingUtilities.invokeLater(() -> panel.setImage(img, false));
        	return;
        };
        drawLogo(logoUrl, panel);

        if (scaleExecutor.isShutdown())
        	scaleExecutor = Executors.newFixedThreadPool(2);
        PENDING.computeIfAbsent(imURL, u ->
          CompletableFuture.supplyAsync(() -> {
            	try {
	              	BufferedImage loaded = ImageIO.read(u);
	              	CACHE.put(u, loaded);
	              	return loaded;
              } catch(Exception | Error e) {
            	  e.printStackTrace();
            	  return getImmediateLoadingSquareLogo();
              }
            }, scaleExecutor)
        );
        PENDING.get(imURL)
        .whenComplete((bi,err)-> {
        	PENDING.remove(imURL);
        	SwingUtilities.invokeLater(() -> panel.setImage(bi, false));
        });
      }
    
    public static void drawLogo(URL url, LogoPanel panel) {
    	if (url == null) {
        	SwingUtilities.invokeLater(() -> panel.setImage(getImmediateLoadingSquareLogo(), true));
        	return;
    	}
        BufferedImage img = CACHE.get(url);
        if (img != null) {
        	SwingUtilities.invokeLater(() -> panel.setImage(img, false));
        	return;
        };


        if (scaleExecutor.isShutdown())
        	scaleExecutor = Executors.newFixedThreadPool(2);
        PENDING.computeIfAbsent(url, u ->
          CompletableFuture.supplyAsync(() -> {
            	try {
	              	BufferedImage loaded = ImageIO.read(u);
	              	CACHE.put(u, loaded);
	              	return loaded;
              } catch(Exception | Error e) {
            	  e.printStackTrace();
            	  return getImmediateLoadingSquareLogo();
              }
            }, scaleExecutor)
        );
        PENDING.get(url)
        .whenComplete((bi,err)-> {
        	PENDING.remove(url);
        	SwingUtilities.invokeLater(() -> panel.setImage(bi, false));
        });
    	SwingUtilities.invokeLater(() -> panel.setImage(getImmediateLoadingSquareLogo(), true));
      }
    
    protected static void setIconPath(String iconPath) {
    	DIJ_ICON_PATH = iconPath;
    	getSoftwareLogo();
    }

    public static ImageIcon getDefaultIcon(int width, int height) {
    	if (DIJ_ICON_PATH == null)
    		return null;
        URL defaultIconUrl = Gui.class.getClassLoader().getResource(DIJ_ICON_PATH);
        if (defaultIconUrl == null) {
            return null;
        }
        // TODO remove BufferedImage defaultImage = ImageIO.read(defaultIconUrl);
        // TODO remove Image scaledDefaultImage = defaultImage.getScaledInstance(width, height, Image.SCALE_FAST);
        Image scaledDefaultImage = MASTER_IMAGE.getScaledInstance(width, height, Image.SCALE_FAST);
        return new ImageIcon(scaledDefaultImage);
    }
    
    

    public static ImageIcon getLoadingIcon(int width, int height) {
        Dimension size = new Dimension(width, height);
        
        // Check if already cached
        ImageIcon cached = ICONS_CACHE.get(size);
        if (cached != null) {
            return cached;
        }
        
        // Check if already being processed
        CompletableFuture<ImageIcon> pending = PENDING_ICONS.get(size);
        if (pending == null && !scaleExecutor.isShutdown()) {
            // Start new scaling operation
            pending = CompletableFuture.supplyAsync(() -> {
                Image scaledImage = MASTER_IMAGE.getScaledInstance(width, height, Image.SCALE_SMOOTH);
                ImageIcon icon = new ImageIcon(scaledImage);
                ICONS_CACHE.put(size, icon);
                PENDING_ICONS.remove(size);
                return icon;
            }, scaleExecutor);
            PENDING_ICONS.put(size, pending);
        }
        
        return createTransparentIcon(width, height);
    }
    
    private static ImageIcon createTransparentIcon(int width, int height) {
        return new ImageIcon(new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB));
    }
    
    // Create a simple placeholder instantly
    /**
     * TODO test if this or {@link #createTransparentIcon(int, int)} is faster
     * @param width
     * @param height
     * @return
     */
    private static ImageIcon createPlaceholderIcon(int width, int height) {
        BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        // Optionally draw something simple like a border or loading text
        Graphics2D g2d = bi.createGraphics();
        g2d.setColor(Color.LIGHT_GRAY);
        g2d.drawRect(0, 0, width - 1, height - 1);
        g2d.dispose();
        return new ImageIcon(bi);
    }
    
    
    // For components that want to update when the exact size is ready
    public static void getLoadingIconWithCallback(int width, int height, Consumer<ImageIcon> callback) {
        ImageIcon immediate = getLoadingIcon(width, height);
        Dimension size = new Dimension(width, height);
        
        CompletableFuture<ImageIcon> pending = PENDING_ICONS.get(size);
        if (pending != null && !scaleExecutor.isShutdown()) {
            pending.thenAcceptAsync(icon -> {
                SwingUtilities.invokeLater(() -> callback.accept(icon));
            }, scaleExecutor);
        }
        
        // Return immediate result
        callback.accept(immediate);
    }
    
    public static void closeThreads() {
    	if (scaleExecutor != null)
    		scaleExecutor.shutdown();
    }
}
