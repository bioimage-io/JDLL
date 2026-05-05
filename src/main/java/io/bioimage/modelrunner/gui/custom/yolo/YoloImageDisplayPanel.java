/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.gui.custom.yolo;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.Timer;
import javax.swing.border.LineBorder;

import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

public class YoloImageDisplayPanel extends JPanel {

    private static final long serialVersionUID = 2304277744020621731L;

    private static final double MIN_ZOOM = 0.2;
    private static final double MAX_ZOOM = 16.0;
    private static final double ZOOM_STEP = 1.1;
    private static final int BOX_STROKE = 2;
    private static final Color BOX_COLOR = new Color(80, 220, 120);
    private static final Color ACTIVE_BOX_COLOR = new Color(255, 166, 0);
    private static final Color EMPTY_BG = new Color(245, 245, 245);
    private static final Color HELP_TEXT = new Color(110, 110, 110);
    private static final int VIEW_PAD = 4;
    private static final double OVERLAY_BUTTON_SIZE_RATIO = 0.09;
    private static final int OVERLAY_BUTTON_MIN = 18;
    private static final int OVERLAY_BUTTON_MAX = 30;
    private static final String EXPAND_SYMBOL = "\u2199\u2197";
    private static final String CONTRACT_SYMBOL = "\u2197\u2199";
    private static final int HINT_HOLD_MS = 1500;
    private static final int HINT_FADE_MS = 1500;
    private static final int HINT_TIMER_DELAY_MS = 40;
    private static final String DEFAULT_HINT_LINE_1 = "Ctrl + wheel for zooming in/out";
    private static final String DEFAULT_HINT_LINE_2 = "Click and move the mouse for panning";

    private RandomAccessibleInterval<?> rai;
    private BufferedImage previewImage;
    private String title;
    private double zoom = 1.0;
    private boolean expandedToFill;
    private boolean drawEnabled;
    private Rectangle imageDrawArea = new Rectangle();
    private Rectangle currentImageRect = new Rectangle();
    private double panX;
    private double panY;
    private Point dragStartScreen;
    private Point panStartScreen;
    private double panStartX;
    private double panStartY;
    private Rectangle2D.Double activeBox;
    private final List<Rectangle2D.Double> boxes = new ArrayList<Rectangle2D.Double>();
    private String emptyMessage = "Preview will appear here";
    private Color boxColor = BOX_COLOR;
    private final JButton expandButton = new JButton(EXPAND_SYMBOL);
    private final Timer hintFadeTimer;
    private long hintShownAt;
    private float hintAlpha;

    protected YoloImageDisplayPanel() {
        setLayout(null);
        setBorder(new LineBorder(Color.GRAY));
        setBackground(YoloUiUtils.INPUT_BG);
        setOpaque(true);
        updateToolTip();
        YoloUiUtils.styleFlatSecondaryButton(expandButton);
        expandButton.setToolTipText("Expand image to fill preview");
        expandButton.addActionListener(e -> setExpandedToFill(!expandedToFill));
        add(expandButton);
        hintFadeTimer = new Timer(HINT_TIMER_DELAY_MS, e -> updateHintAlpha());
        hintFadeTimer.setRepeats(true);

        MouseAdapter mouseAdapter = new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (previewImage == null || !currentImageRect.contains(e.getPoint())) {
                    return;
                }
                if (!drawEnabled) {
                    panStartScreen = e.getPoint();
                    panStartX = panX;
                    panStartY = panY;
                    return;
                }
                dragStartScreen = e.getPoint();
                activeBox = toImageRectangle(dragStartScreen, dragStartScreen);
                repaint();
            }

            @Override
            public void mouseDragged(MouseEvent e) {
                if (previewImage == null) {
                    return;
                }
                if (!drawEnabled) {
                    if (panStartScreen == null) {
                        return;
                    }
                    panX = panStartX + e.getX() - panStartScreen.x;
                    panY = panStartY + e.getY() - panStartScreen.y;
                    currentImageRect = computeCurrentImageRect(computeImageDrawArea());
                    repaint();
                    return;
                }
                if (dragStartScreen == null) {
                    return;
                }
                activeBox = toImageRectangle(dragStartScreen, e.getPoint());
                repaint();
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                if (!drawEnabled) {
                    panStartScreen = null;
                    repaint();
                    return;
                }
                if (dragStartScreen == null || activeBox == null) {
                    dragStartScreen = null;
                    activeBox = null;
                    repaint();
                    return;
                }
                if (activeBox.width > 1 && activeBox.height > 1) {
                    boxes.add(activeBox);
                }
                dragStartScreen = null;
                activeBox = null;
                repaint();
            }

            @Override
            public void mouseWheelMoved(MouseWheelEvent e) {
                if (!e.isControlDown()) {
                    return;
                }
                if (previewImage == null) {
                    return;
                }
                imageDrawArea = computeImageDrawArea();
                Rectangle oldRect = computeCurrentImageRect(imageDrawArea);
                double imageX = previewImage.getWidth() / 2.0;
                double imageY = previewImage.getHeight() / 2.0;
                if (oldRect.width > 0 && oldRect.height > 0 && oldRect.contains(e.getPoint())) {
                    imageX = (e.getX() - oldRect.x) * previewImage.getWidth() / (double) oldRect.width;
                    imageY = (e.getY() - oldRect.y) * previewImage.getHeight() / (double) oldRect.height;
                }
                double nextZoom = e.getWheelRotation() < 0 ? zoom * ZOOM_STEP : zoom / ZOOM_STEP;
                zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, nextZoom));
                updatePanForAnchor(e.getPoint(), imageX, imageY);
                repaint();
            }
        };
        addMouseListener(mouseAdapter);
        addMouseMotionListener(mouseAdapter);
        addMouseWheelListener(mouseAdapter);
    }

    public void setDrawEnabled(boolean drawEnabled) {
        this.drawEnabled = drawEnabled;
        updateToolTip();
    }

    public boolean isDrawEnabled() {
        return drawEnabled;
    }

    public void clearBoxes() {
        boxes.clear();
        activeBox = null;
        repaint();
    }

    public List<Rectangle2D.Double> getBoxes() {
        return new ArrayList<Rectangle2D.Double>(boxes);
    }

    public void clearImage() {
        this.rai = null;
        this.previewImage = null;
        this.title = null;
        this.zoom = 1.0;
        this.expandedToFill = false;
        this.panX = 0.0;
        this.panY = 0.0;
        this.dragStartScreen = null;
        this.panStartScreen = null;
        this.activeBox = null;
        this.imageDrawArea = new Rectangle();
        this.currentImageRect = new Rectangle();
        this.boxes.clear();
        this.hintAlpha = 0.0f;
        this.hintFadeTimer.stop();
        repaint();
    }

    public void setEmptyMessage(String emptyMessage) {
        this.emptyMessage = emptyMessage == null || emptyMessage.trim().isEmpty()
                ? "Preview will appear here"
                : emptyMessage.trim();
        repaint();
    }

    public <T extends RealType<T> & NativeType<T>> void setImage(RandomAccessibleInterval<T> rai, String title) {
        this.rai = rai;
        this.title = title;
        this.previewImage = buildPreview(rai);
        this.zoom = 1.0;
        this.expandedToFill = false;
        this.panX = 0.0;
        this.panY = 0.0;
        this.panStartScreen = null;
        this.boxColor = BOX_COLOR;
        clearBoxes();
        updateExpandButtonState();
        showDefaultHint();
        repaint();
    }

    public void setBufferedImage(BufferedImage image, String title) {
        this.rai = null;
        this.title = title;
        this.previewImage = image;
        this.zoom = 1.0;
        this.expandedToFill = false;
        this.drawEnabled = false;
        this.panX = 0.0;
        this.panY = 0.0;
        this.dragStartScreen = null;
        this.panStartScreen = null;
        this.activeBox = null;
        this.boxes.clear();
        this.boxColor = BOX_COLOR;
        updateToolTip();
        updateExpandButtonState();
        if (image != null) {
            showDefaultHint();
        }
        repaint();
    }

    public void setReadOnlyBoxes(List<Rectangle2D.Double> boxes, Color color) {
        this.drawEnabled = false;
        this.boxes.clear();
        if (boxes != null) {
            this.boxes.addAll(boxes);
        }
        this.boxColor = color == null ? BOX_COLOR : color;
        updateToolTip();
        repaint();
    }

    public void setExpandedToFill(boolean expandedToFill) {
        this.expandedToFill = expandedToFill;
        this.panX = 0.0;
        this.panY = 0.0;
        updateExpandButtonState();
        repaint();
    }

    public boolean isExpandedToFill() {
        return expandedToFill;
    }

    private void updateToolTip() {
        String dragHelp = drawEnabled
                ? "Click and move the mouse to draw rectangle"
                : "Click and move the mouse for panning";
        setToolTipText("<html>Ctrl + wheel for zooming in/out<br>" + dragHelp + "</html>");
    }

    private void updateExpandButtonState() {
        expandButton.setText(expandedToFill ? CONTRACT_SYMBOL : EXPAND_SYMBOL);
        expandButton.setToolTipText(expandedToFill ? "Contract image to fit preview" : "Expand image to fill preview");
    }

    @Override
    public void doLayout() {
        int size = Math.max(OVERLAY_BUTTON_MIN,
                Math.min(OVERLAY_BUTTON_MAX, (int) Math.round(Math.min(getWidth(), getHeight()) * OVERLAY_BUTTON_SIZE_RATIO)));
        expandButton.setBounds(getWidth() - VIEW_PAD - size, VIEW_PAD, size, size);
        expandButton.setFont(expandButton.getFont().deriveFont((float) Math.max(YoloUiUtils.MIN_FONT_SIZE,
                Math.min(YoloUiUtils.MAX_CONTROL_FONT_SIZE - 2, size * 0.42f))));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        if (previewImage == null) {
            g2.setColor(EMPTY_BG);
            g2.fillRect(0, 0, getWidth(), getHeight());
            g2.setColor(HELP_TEXT);
            drawCenteredMessage(g2, emptyMessage);
            g2.dispose();
            return;
        }

        imageDrawArea = computeImageDrawArea();
        currentImageRect = computeCurrentImageRect(imageDrawArea);
        int drawnW = currentImageRect.width;
        int drawnH = currentImageRect.height;
        int x = currentImageRect.x;
        int y = currentImageRect.y;

        g2.setClip(imageDrawArea);
        g2.drawImage(previewImage, x, y, drawnW, drawnH, null);
        drawBoxes(g2, x, y, drawnW, drawnH);
        g2.setClip(null);

        if (title != null && !title.isEmpty()) {
            g2.setColor(HELP_TEXT);
            g2.drawString(title, imageDrawArea.x + 6, imageDrawArea.y + 16);
        }
        paintHintOverlay(g2);
        g2.dispose();
    }

    private void showDefaultHint() {
        hintShownAt = System.currentTimeMillis();
        hintAlpha = 1.0f;
        hintFadeTimer.restart();
    }

    private void updateHintAlpha() {
        long elapsed = System.currentTimeMillis() - hintShownAt;
        if (elapsed <= HINT_HOLD_MS) {
            hintAlpha = 1.0f;
        } else {
            float fadeProgress = (elapsed - HINT_HOLD_MS) / (float) HINT_FADE_MS;
            hintAlpha = Math.max(0.0f, 1.0f - fadeProgress);
        }
        if (hintAlpha <= 0.0f) {
            hintFadeTimer.stop();
        }
        repaint();
    }

    private void paintHintOverlay(Graphics2D g2) {
        if (hintAlpha <= 0.0f || previewImage == null) {
            return;
        }
        Graphics2D overlay = (Graphics2D) g2.create();
        overlay.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        overlay.setComposite(java.awt.AlphaComposite.getInstance(java.awt.AlphaComposite.SRC_OVER, hintAlpha));
        int fontSize = Math.max(11, Math.min(16, getHeight() / 20));
        Font font = overlay.getFont().deriveFont(Font.BOLD, (float) fontSize);
        overlay.setFont(font);
        FontMetrics fm = overlay.getFontMetrics();
        int textW = Math.max(fm.stringWidth(DEFAULT_HINT_LINE_1), fm.stringWidth(DEFAULT_HINT_LINE_2));
        int padX = 16;
        int padY = 10;
        int lineGap = 4;
        int lineH = fm.getHeight();
        int boxW = textW + 2 * padX;
        int boxH = 2 * lineH + lineGap + 2 * padY;
        int boxX = (getWidth() - boxW) / 2;
        int boxY = (getHeight() - boxH) / 2;

        overlay.setColor(new Color(255, 255, 255, 210));
        overlay.fillRoundRect(boxX, boxY, boxW, boxH, 14, 14);
        overlay.setColor(new Color(120, 128, 144, 180));
        overlay.drawRoundRect(boxX, boxY, boxW, boxH, 14, 14);
        overlay.setColor(new Color(70, 78, 98));
        drawCenteredOverlayLine(overlay, DEFAULT_HINT_LINE_1, boxX, boxY + padY, boxW, lineH);
        drawCenteredOverlayLine(overlay, DEFAULT_HINT_LINE_2, boxX, boxY + padY + lineH + lineGap, boxW, lineH);
        overlay.dispose();
    }

    private static void drawCenteredOverlayLine(Graphics2D g2, String text, int x, int y, int width, int height) {
        FontMetrics fm = g2.getFontMetrics();
        int tx = x + (width - fm.stringWidth(text)) / 2;
        int ty = y + ((height - fm.getHeight()) / 2) + fm.getAscent();
        g2.drawString(text, tx, ty);
    }

    private static void drawCenteredMessage(Graphics2D g2, String text) {
        FontMetrics fm = g2.getFontMetrics();
        Rectangle clip = g2.getClipBounds();
        if (clip == null) {
            return;
        }
        String message = text == null ? "" : text;
        int tx = clip.x + Math.max(8, (clip.width - fm.stringWidth(message)) / 2);
        int ty = clip.y + Math.max(fm.getAscent(), (clip.height - fm.getHeight()) / 2 + fm.getAscent());
        g2.drawString(message, tx, ty);
    }

    private void drawBoxes(Graphics2D g2, int drawX, int drawY, int drawW, int drawH) {
        double sx = drawW / (double) previewImage.getWidth();
        double sy = drawH / (double) previewImage.getHeight();

        g2.setStroke(new BasicStroke(BOX_STROKE));
        g2.setColor(boxColor);
        for (Rectangle2D.Double box : boxes) {
            g2.draw(new Rectangle2D.Double(drawX + box.x * sx, drawY + box.y * sy, box.width * sx, box.height * sy));
        }
        if (activeBox != null) {
            g2.setColor(ACTIVE_BOX_COLOR);
            g2.draw(new Rectangle2D.Double(drawX + activeBox.x * sx, drawY + activeBox.y * sy,
                    activeBox.width * sx, activeBox.height * sy));
        }
    }

    private Rectangle computeImageDrawArea() {
        int panelW = Math.max(1, getWidth() - 2 * VIEW_PAD);
        int panelH = Math.max(1, getHeight() - 2 * VIEW_PAD);
        return new Rectangle((getWidth() - panelW) / 2, (getHeight() - panelH) / 2, panelW, panelH);
    }

    private Rectangle computeCurrentImageRect(Rectangle area) {
        Rectangle baseRect = computeBaseImageRect(area);
        int drawnW = Math.max(1, (int) Math.round(baseRect.width * zoom));
        int drawnH = Math.max(1, (int) Math.round(baseRect.height * zoom));
        int centeredX = area.x + (area.width - drawnW) / 2;
        int centeredY = area.y + (area.height - drawnH) / 2;
        int x = centeredX + (int) Math.round(panX);
        int y = centeredY + (int) Math.round(panY);

        if (drawnW <= area.width) {
            x = centeredX;
            panX = 0.0;
        } else {
            int minX = area.x + area.width - drawnW;
            int maxX = area.x;
            x = Math.max(minX, Math.min(maxX, x));
            panX = x - centeredX;
        }
        if (drawnH <= area.height) {
            y = centeredY;
            panY = 0.0;
        } else {
            int minY = area.y + area.height - drawnH;
            int maxY = area.y;
            y = Math.max(minY, Math.min(maxY, y));
            panY = y - centeredY;
        }
        return new Rectangle(x, y, drawnW, drawnH);
    }

    private Rectangle computeBaseImageRect(Rectangle area) {
        double imgRatio = previewImage.getWidth() / (double) previewImage.getHeight();
        int drawW;
        int drawH;
        if (!expandedToFill) {
            drawW = area.width;
            drawH = (int) Math.round(drawW / imgRatio);
            if (drawH > area.height) {
                drawH = area.height;
                drawW = (int) Math.round(drawH * imgRatio);
            }
        } else {
            drawW = area.width;
            drawH = (int) Math.round(drawW / imgRatio);
            if (drawH < area.height) {
                drawH = area.height;
                drawW = (int) Math.round(drawH * imgRatio);
            }
        }
        int x = area.x + (area.width - drawW) / 2;
        int y = area.y + (area.height - drawH) / 2;
        return new Rectangle(x, y, Math.max(1, drawW), Math.max(1, drawH));
    }

    private void updatePanForAnchor(Point anchorPoint, double imageX, double imageY) {
        Rectangle area = computeImageDrawArea();
        int drawnW = Math.max(1, (int) Math.round(area.width * zoom));
        int drawnH = Math.max(1, (int) Math.round(area.height * zoom));
        int centeredX = area.x + (area.width - drawnW) / 2;
        int centeredY = area.y + (area.height - drawnH) / 2;

        double targetX = anchorPoint.x - imageX * drawnW / (double) previewImage.getWidth();
        double targetY = anchorPoint.y - imageY * drawnH / (double) previewImage.getHeight();
        panX = targetX - centeredX;
        panY = targetY - centeredY;
        currentImageRect = computeCurrentImageRect(area);
    }

    private Rectangle2D.Double toImageRectangle(Point start, Point end) {
        Rectangle area = currentImageRect;
        int minX = Math.max(area.x, Math.min(start.x, end.x));
        int minY = Math.max(area.y, Math.min(start.y, end.y));
        int maxX = Math.min(area.x + area.width, Math.max(start.x, end.x));
        int maxY = Math.min(area.y + area.height, Math.max(start.y, end.y));
        double relX1 = (minX - area.x) / (double) area.width;
        double relY1 = (minY - area.y) / (double) area.height;
        double relX2 = (maxX - area.x) / (double) area.width;
        double relY2 = (maxY - area.y) / (double) area.height;
        return new Rectangle2D.Double(
                relX1 * previewImage.getWidth(),
                relY1 * previewImage.getHeight(),
                Math.max(0, (relX2 - relX1) * previewImage.getWidth()),
                Math.max(0, (relY2 - relY1) * previewImage.getHeight()));
    }

    private static <T extends RealType<T> & NativeType<T>> BufferedImage buildPreview(RandomAccessibleInterval<T> source) {
        RandomAccessibleInterval<T> rai = source;
        while (rai.numDimensions() > 3) {
            rai = Views.hyperSlice(rai, rai.numDimensions() - 1, 0);
        }
        if (rai.numDimensions() == 3 && rai.dimension(2) != 3 && rai.dimension(2) != 4) {
            rai = Views.hyperSlice(rai, 2, 0);
        }
        if (rai.numDimensions() == 2) {
            return buildGrayPreview(rai);
        } else if (rai.numDimensions() == 3) {
            return buildColorPreview(rai);
        }
        return new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
    }

    private static <T extends RealType<T> & NativeType<T>> BufferedImage buildGrayPreview(RandomAccessibleInterval<T> rai) {
        int width = (int) rai.dimension(0);
        int height = (int) rai.dimension(1);
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        double[] range = findRange(rai);
        double min = range[0];
        double max = range[1];
        if (max <= min) {
            max = min + 1;
        }
        RandomAccess<T> access = rai.randomAccess();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                access.setPosition(x, 0);
                access.setPosition(y, 1);
                int v = scaleToByte(access.get().getRealDouble(), min, max);
                int rgb = (v << 16) | (v << 8) | v;
                image.setRGB(x, y, rgb);
            }
        }
        return image;
    }

    private static <T extends RealType<T> & NativeType<T>> BufferedImage buildColorPreview(RandomAccessibleInterval<T> rai) {
        int width = (int) rai.dimension(0);
        int height = (int) rai.dimension(1);
        int channels = (int) rai.dimension(2);
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        double[] range = findRange(rai);
        double min = range[0];
        double max = range[1];
        if (max <= min) {
            max = min + 1;
        }
        RandomAccess<T> access = rai.randomAccess();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int[] rgb = new int[] {0, 0, 0};
                for (int c = 0; c < Math.min(3, channels); c++) {
                    access.setPosition(x, 0);
                    access.setPosition(y, 1);
                    access.setPosition(c, 2);
                    rgb[c] = scaleToByte(access.get().getRealDouble(), min, max);
                }
                image.setRGB(x, y, (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]);
            }
        }
        return image;
    }

    private static <T extends RealType<T> & NativeType<T>> double[] findRange(RandomAccessibleInterval<T> rai) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (T value : Views.iterable(rai)) {
            double v = value.getRealDouble();
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
        }
        if (!Double.isFinite(min) || !Double.isFinite(max)) {
            return new double[] {0, 1};
        }
        return new double[] {min, max};
    }

    private static int scaleToByte(double value, double min, double max) {
        double norm = (value - min) / (max - min);
        norm = Math.max(0, Math.min(1, norm));
        return (int) Math.round(norm * 255.0);
    }
}
