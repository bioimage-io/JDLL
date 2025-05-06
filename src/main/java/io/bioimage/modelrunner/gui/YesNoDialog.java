/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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
