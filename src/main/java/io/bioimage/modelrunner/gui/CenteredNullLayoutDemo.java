package io.bioimage.modelrunner.gui;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class CenteredNullLayoutDemo {
    /** A red square that stays square and resizes. */
    static class LogoPanel extends JPanel {
        LogoPanel() { setBackground(Color.RED); }
        @Override protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            int side = Math.min(getWidth(), getHeight());
            g.setColor(Color.RED);
            g.fillRect(0, 0, side, side);
        }
    }

    private static JLabel title;
    private static JLabel subtitle;

    private static void createAndShow() {
        JFrame frame = new JFrame("Centered UI Demo");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(null);

        // 1) Transparent empty panel at far left
        JPanel empty = new JPanel();
        empty.setOpaque(false);
        frame.add(empty);

        // 2) Red square “logo” to the right of empty
        LogoPanel logo = new LogoPanel();
        frame.add(logo);

        // 3) The two labels
        title = new JLabel("deepIcy");
        title.setFont(title.getFont().deriveFont(Font.BOLD, 28f));
        frame.add(title);

        subtitle = new JLabel("The Icy plugin for AI");
        subtitle.setFont(subtitle.getFont().deriveFont(Font.PLAIN, 16f));
        frame.add(subtitle);

        // 4) Progress bar right of subtitle/title
        JProgressBar bar = new JProgressBar(0, 100);
        bar.setStringPainted(true);
        bar.setPreferredSize(new Dimension(150, 24));
        frame.add(bar);

        // 5) On resize, reposition everything
        frame.addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                Insets in = frame.getInsets();
                int W = frame.getWidth()  - in.left - in.right;
                int H = frame.getHeight() - in.top  - in.bottom;
                
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
                

                title.setFont(title.getFont().deriveFont(Font.PLAIN, fontSize));
                subtitle.setFont(subtitle.getFont().deriveFont(Font.PLAIN, sFontSize));

                // measure text
                Dimension tSz = title.getPreferredSize();
                Dimension sSz = subtitle.getPreferredSize();
                int remainingPixels = H - tSz .height - sSz.height;
                int titleGap = Math.max(1,  remainingPixels / 6);
                int headerTop = Math.max(0, (remainingPixels - titleGap) / 2);

                // center each label independently
                int xTitle    = (W - tSz.width) / 2;
                int xSubtitle = (W - sSz.width) / 2;
                int yTitle    = headerTop;
                int ySubtitle = headerTop + tSz.height + titleGap;

                // position title & subtitle
                title  .setBounds(xTitle,    yTitle,    tSz.width, tSz.height);
                subtitle.setBounds(xSubtitle, ySubtitle, sSz.width, sSz.height);

                int gap = 8;  // between text and bar
                // position progress bar next to the widest text
                int textW = Math.max(xTitle + tSz.width, xSubtitle + sSz.width);
                int xBar = textW + gap;
                int barH = bar.getPreferredSize().height;
                int yBar = headerTop ;
                bar.setBounds(xBar, yBar, bar.getPreferredSize().width, barH);

                // compute logo size so it never intrudes past the title
                int minLogoW = Math.min(xTitle, xSubtitle);
                int logoSide = Math.min(H, minLogoW);
                int xLogo = minLogoW - logoSide;
                if (xLogo < 0) xLogo = 0;  // guard against super‑narrow windows

                // place empty panel from x=0 up to logo start
                empty.setBounds(0, 0, xLogo, H);

                // place logo immediately to its right
                logo.setBounds(xLogo, 0, logoSide, logoSide);
            }
        });

        frame.setSize(800, 220);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(CenteredNullLayoutDemo::createAndShow);
    }
}

