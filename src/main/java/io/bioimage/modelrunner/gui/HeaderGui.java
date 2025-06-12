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

import java.awt.Dimension;
import java.awt.Font;
import java.awt.Insets;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.net.URL;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;


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
}

