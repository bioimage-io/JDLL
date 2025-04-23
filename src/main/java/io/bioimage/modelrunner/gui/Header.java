package io.bioimage.modelrunner.gui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.io.File;
import java.net.URL;
import java.util.function.Consumer;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.border.LineBorder;

import io.bioimage.modelrunner.gui.adapter.GuiAdapter;

public class Header extends JPanel {
    private static final long serialVersionUID = -7691139174208436363L;


    // pulled out so we can resize them on‐the‐fly:
    private final JLabel titleLabel;
    private final JLabel subtitleLabel;

    private JProgressBar progressBar;
    private JLabel progressLabel;

    public Header(GuiAdapter adapter) {
        super(new BorderLayout());
        setBackground(Color.GRAY);
        setBorder(new LineBorder(Color.BLACK, 2, true));

        // — create + style the two text labels
        titleLabel    = new JLabel(adapter.getSoftwareName(), SwingConstants.CENTER);
        titleLabel.setForeground(Color.WHITE);

        subtitleLabel = new JLabel(adapter.getSoftwareDescription(), SwingConstants.CENTER);
        subtitleLabel.setForeground(Color.WHITE);
        
        createProgressBar();
        URL defaultIconUrl = Header.class.getClassLoader().getResource(adapter.getIconPath());
        this.add(new HeaderGui(titleLabel, subtitleLabel, progressBar, progressLabel, defaultIconUrl), BorderLayout.CENTER);
    }

    private void createProgressBar() {
        progressBar = new JProgressBar(0, 100);
        progressBar.setStringPainted(false);
        progressBar.setBackground(Color.LIGHT_GRAY);
        progressBar.setForeground(new Color(46, 204, 113));

        progressLabel = new JLabel("Processing...", SwingConstants.CENTER);
        progressLabel.setBackground(Color.GRAY);
        progressLabel.setOpaque(true);
    }
	
	protected void setGUIStartInstallation() {
    	SwingUtilities.invokeLater(() -> {
    		progressBar.setIndeterminate(true);
            progressBar.setStringPainted(true);
    		progressBar.setVisible(true);
    		progressBar.setString("0%");
    		progressLabel.setText("Preparing installation...");
    		progressLabel.setVisible(true);

    	});
	}
	
	public Consumer<String> createStringConsumer(){
		Consumer<String> consumer = (ss) -> {
			SwingUtilities.invokeLater(() -> {
    			this.progressLabel.setText("Installing " + new File(ss).getName());
			});
		};
		return consumer;
	}
	
	public Consumer<Double> createProgressConsumer(){
		Consumer<Double> consumer = (dd) -> {
			SwingUtilities.invokeLater(() -> {
				if (progressBar.isIndeterminate())
					progressBar.setIndeterminate(false);
				double perc = Math.floor(dd * 1000) / 10;
				progressBar.setString(perc + "%");
	    		progressBar.setValue((int) perc);
	    		if (perc == 100){
	        		progressLabel.setVisible(false);
	        		progressBar.setVisible(false);
	    		}
			});
		};
		return consumer;
	}
}
