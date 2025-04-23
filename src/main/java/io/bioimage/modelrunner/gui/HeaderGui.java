package io.bioimage.modelrunner.gui;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;

import javax.imageio.ImageIO;
import javax.swing.*;

public class CenteredNullLayoutDemo {
    /** A red square that stays square and resizes. */
    static class LogoPanel extends JPanel {
    	private static final long serialVersionUID = -3161345822406354L;
		private BufferedImage image;
        LogoPanel(BufferedImage img) { 
        	this.image = img;
        	setOpaque(false); 
        }
        @Override protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            int side = Math.min(getWidth(), getHeight());
            if (image != null) {
                g.drawImage(image, 0, 0, side, side, this);
            } else {
                g.fillRect(0, 0, side, side);
            }
        }
        public void setImage(BufferedImage image) {
        	this.image = image;
            repaint();
        }
    }

    private static JLabel title;
    private static JLabel subtitle;
    private static JLabel barSubtitle;
    private static JProgressBar bar;

    private static void createAndShow() throws IOException {
        JFrame frame = new JFrame("Centered UI Demo");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(null);

        // 1) Transparent empty panel at far left
        JPanel empty = new JPanel();
        empty.setOpaque(false);
        frame.add(empty);

        // 2) Red square “logo” to the right of empty
        String str = "/home/carlos/git/deepimagej-plugin/src/main/resources/dij_imgs/deepimagej_icon.png";
        BufferedImage logoImg = ImageIO.read(new File(str));
        LogoPanel logo = new LogoPanel(null);
        new SwingWorker<BufferedImage, Void>() {
            @Override
            protected BufferedImage doInBackground() throws Exception {
                // simulate slowness...
                 Thread.sleep(2000); 
                // load your real image (from disk, network, classpath…)
                return ImageIO.read(new File(str));
            }
            @Override
            protected void done() {
                try {
					logo.setImage(get());
				} catch (InterruptedException | ExecutionException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
            }
        }.execute();
        frame.add(logo);

        // 3) The two labels
        title = new JLabel("deepIcy");
        title.setFont(title.getFont().deriveFont(Font.BOLD, 28f));
        frame.add(title);

        subtitle = new JLabel("The Icy plugin for AI");
        subtitle.setFont(subtitle.getFont().deriveFont(Font.PLAIN, 16f));
        frame.add(subtitle);

        // 4) Progress bar right of subtitle/title
        bar = new JProgressBar(0, 100);
        bar.setStringPainted(true);
        bar.setVisible(false);
        frame.add(bar);
        barSubtitle = new JLabel("Gteakefnkjbgvrekjih vjwkrsnvkjrebhr jrvbksrvb");
        barSubtitle.setVisible(false);
        frame.add(barSubtitle);

        // 5) On resize, reposition everything
        frame.addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                Insets in = frame.getInsets();
                int W = frame.getWidth()  - in.left - in.right;
                int H = frame.getHeight() - in.top  - in.bottom;
                
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
                bar.setBounds(xBar, yBar, (W - textW) - barGap * 2, barH);

                int yString = titleGap / 2 +  (yTitle + ySubtitle + tSz.height) / 2;
                barSubtitle.setFont(barSubtitle.getFont().deriveFont(Font.PLAIN, sFontSize * 0.6f));
                barSubtitle.setBounds(xBar, yString, (W - textW) - barGap * 2, barH);

                // compute logo size so it never intrudes past the title
                int logoInsetX = Math.max(logoInset, logoSize / 10);
                int xLogo = xSubtitle - logoSize - logoInsetX;
                if (xLogo < 0) xLogo = 0;  // guard against super‑narrow windows

                // place empty panel from x=0 up to logo start
                empty.setBounds(0, 0, xLogo, H);

                // place logo immediately to its right
                int logoInsetY = Math.max(logoInset, (H - logoSize) / 2);
                logo.setBounds(xLogo, logoInsetY, logoSize, logoSize);
            }
        });

        frame.setSize(800, 220);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
			try {
				createAndShow();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		});
    }
}

