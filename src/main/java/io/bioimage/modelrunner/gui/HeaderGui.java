package io.bioimage.modelrunner.gui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Insets;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.Map;

import javax.swing.*;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.gui.adapter.GuiAdapter;
import io.bioimage.modelrunner.gui.adapter.RunnerAdapter;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class HeaderGui extends JPanel {
    private static final long serialVersionUID = -306110026903658536L;


    protected final JLabel title;
    protected final JLabel subtitle;
    protected final JLabel barSubtitle;
    protected final JProgressBar bar;
    protected final URL logoURL;
    
    protected HeaderGui(JLabel title, JLabel subtitle, JProgressBar bar, JLabel barSubtitle, URL logoURL) {
    	super(null);
    	this.bar = bar;
    	this.title = title;
    	this.subtitle = subtitle;
    	this.barSubtitle = barSubtitle;
    	this.logoURL = logoURL;
    	createAndShow();
    }

    private void createAndShow() {

        // 1) Transparent empty panel at far left
        JPanel empty = new JPanel();
        empty.setOpaque(false);
        add(empty);

        // 2) Red square “logo” to the right of empty
        LogoPanel logo = new LogoPanel();
        add(logo);
        DefaultIcon.drawLogo(logoURL, logo);

        // 3) The two labels
        title.setFont(title.getFont().deriveFont(Font.BOLD, 28f));
        add(title);

        subtitle.setFont(subtitle.getFont().deriveFont(Font.PLAIN, 16f));
        add(subtitle);

        // 4) Progress bar right of subtitle/title
        bar.setStringPainted(true);
        bar.setVisible(false);
        add(bar);
        barSubtitle.setVisible(false);
        add(barSubtitle);

        // 5) On resize, reposition everything
        addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                Insets in = getInsets();
                int W = getWidth()  - in.left - in.right;
                int H = getHeight() - in.top  - in.bottom;
                
                int logoInset = 2;
                
                double ratio = W / (double) H;

                float fontSize;
                float sFontSize;
                if (ratio > 6) {
                	fontSize = H / 2.8f;
                	sFontSize = H / 4.9f;
                } else {
                	fontSize = W / 16.5f;
                	sFontSize = W / 28.8f;
                }
                

                title.setFont(title.getFont().deriveFont(Font.BOLD, fontSize));
                subtitle.setFont(subtitle.getFont().deriveFont(Font.PLAIN, sFontSize));

                // measure text
                Dimension tSz = title.getPreferredSize();
                Dimension sSz = subtitle.getPreferredSize();
                int remainingPixels = H - tSz .height - sSz.height;
                int titleGap = Math.max(1,  remainingPixels / 6);
                int headerTop = Math.max(0, (remainingPixels - titleGap) / 2);
                
                int logoSize = Math.min(H - logoInset * 2, sSz.width / 2);
                logoSize = Math.max(1, logoSize);

                // center each label independently
                int xTitle    = (W - tSz.width) / 2;
                int xSubtitle = (W - sSz.width) / 2;
                int yTitle    = headerTop;
                int ySubtitle = headerTop + tSz.height + titleGap;

                // position title & subtitle
                title  .setBounds(xTitle,    yTitle,    tSz.width, tSz.height);
                subtitle.setBounds(xSubtitle, ySubtitle, sSz.width, sSz.height);

                int minBarGap = 2;  // between text and bar
                // position progress bar next to the widest text
                int textW = Math.max(xTitle + tSz.width, xSubtitle + sSz.width);
                int barGap = Math.max(minBarGap, (W - textW) / 10);
                int xBar = textW + barGap;
                int barH = sSz.height;
                int yBar = - barH - titleGap / 2 +  (yTitle + ySubtitle + tSz.height) / 2;
                int barW = Math.max(1, (W - textW) - barGap * 2);
                bar.setBounds(xBar, yBar, barW, barH);

                int yString = titleGap / 2 +  (yTitle + ySubtitle + tSz.height) / 2;
                barSubtitle.setFont(barSubtitle.getFont().deriveFont(Font.PLAIN, sFontSize * 0.6f));
                int barSubtitleW = Math.max(1, (W - textW) - barGap * 2);
                barSubtitle.setBounds(xBar, yString, barSubtitleW, barH);

                // compute logo size so it never intrudes past the title
                int logoInsetX = Math.max(logoInset, logoSize / 10);
                int xLogo = xSubtitle - logoSize - logoInsetX;
                if (xLogo < 1) xLogo = 1;  // guard against super‑narrow windows

                // place empty panel from x=0 up to logo start
                empty.setBounds(0, 0, xLogo, Math.max(H, 1));

                // place logo immediately to its right
                int logoInsetY = Math.max(logoInset, (H - logoSize) / 2);
                logo.setBounds(xLogo, logoInsetY, logoSize, logoSize);
                
            }
        });
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            // 1) Create the frame
            JFrame frame = new JFrame("Header Test");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(300, 400);  // or whatever size you need
            frame.setLocationRelativeTo(null);
            
            GuiAdapter adapter = new GuiAdapter () {

				@Override
				public String getSoftwareName() {
					return "JOHN DOE";
				}
				
				@Override
				public Color getHeaderColor() {
					return Color.gray;
				}

				@Override
				public String getSoftwareDescription() {
					return "The best AI software";
				}

				@Override
				public String getIconPath() {
					return "/home/carlos/git/deep-icy/src/main/resources/deepicy_imgs/icy_logo.png";
				}

				@Override
				public String getModelsDir() {
					return null;
				}

				@Override
				public String getEnginesDir() {
					return null;
				}

				@Override
				public RunnerAdapter createRunner(ModelDescriptor descriptor) throws IOException, LoadEngineException {
					return null;
				}

				@Override
				public RunnerAdapter createRunner(ModelDescriptor descriptor, String enginesPath)
						throws IOException, LoadEngineException {
					return null;
				}

				@Override
				public <T extends RealType<T> & NativeType<T>> void displayRai(RandomAccessibleInterval<T> rai,
						String axesOrder, String imTitle) {
					
				}

				@Override
				public <T extends RealType<T> & NativeType<T>> List<Tensor<T>> getInputTensors(
						ModelDescriptor descriptor) {
					return null;
				}

				@Override
				public List<String> getInputImageNames() {
					return null;
				}

				@Override
				public <T extends RealType<T> & NativeType<T>> List<Tensor<T>> convertToInputTensors(
						Map<String, Object> inputs, ModelDescriptor descriptor) {
					return null;
				}

				@Override
				public Color getTitleColor() {
					return Color.white;
				}

				@Override
				public Color getSubtitleColor() {
					return Color.white;
				}
            	
            };

		    //new File("/home/carlos/git/deep-icy/src/main/resources/deepicy_imgs/icy_logo.png").toURL();
            // 2) Create and configure your card
            Header header = new Header(adapter);

            // 3) Add to frame (since ModelCardGui uses null layout internally,
            //    we’ll use BorderLayout here to have it fill the window)
            frame.getContentPane().setLayout(new BorderLayout());
            frame.getContentPane().add(header, BorderLayout.CENTER);

            // 4) Show it
            frame.setVisible(true);
        });
    }
}

