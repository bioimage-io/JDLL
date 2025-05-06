package io.bioimage.modelrunner.gui;

import java.lang.reflect.InvocationTargetException;

import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;

public class YesNoDialog {
	
    public static boolean askQuestion(String title, String message) {
        // Show the Yes/No dialog
    	int[] response = new int[1];
    	try {
			SwingUtilities.invokeAndWait(() ->{
			    int res = JOptionPane.showConfirmDialog(null, message, title, JOptionPane.YES_NO_OPTION);
			    response[0] = res;
			});
		} catch (InvocationTargetException | InterruptedException e) {
			e.printStackTrace();
		}
        return response[0] == JOptionPane.YES_OPTION ? true : false;
    }
}
