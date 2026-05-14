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
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;

import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.SwingWorker;
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
    private static final double ZOOM_STEP = 1.1;
    private static final double MIN_VISIBLE_SOURCE_SIZE = 64.0;
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
    private static final int MAX_RENDER_SIDE = 1024;
    private static final int MAX_RENDER_PIXELS = MAX_RENDER_SIDE * MAX_RENDER_SIDE;
    private static final int RANGE_SAMPLE_BUDGET = 65536;
    private static final String DEFAULT_HINT_LINE_1 = "Ctrl + wheel for zooming in/out";
    private static final String DEFAULT_HINT_LINE_2 = "Click and move the mouse for panning";

    private PreviewSource previewSource;
    private BufferedImage renderedViewport;
    private RenderRequest renderedRequest;
    private RenderRequest pendingRequest;
    private SwingWorker<RenderResult, Void> renderWorker;
    private long sourceVersion;
    private long renderVersion;
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
    private Point zoomAnchorScreen;
    private double zoomAnchorImageX;
    private double zoomAnchorImageY;
    private Rectangle2D.Double activeBox;
    private final List<Rectangle2D.Double> boxes = new ArrayList<Rectangle2D.Double>();
    private String emptyMessage = "Preview will appear here";
    private Color boxColor = BOX_COLOR;
    private Color titleColor = HELP_TEXT;
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
                clearZoomAnchor();
                updateCurrentImageRect();
                if (previewSource == null || !currentImageRect.contains(e.getPoint())) {
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
                clearZoomAnchor();
                if (previewSource == null) {
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
                clearZoomAnchor();
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
                if (previewSource == null) {
                    return;
                }
                e.consume();
                imageDrawArea = computeImageDrawArea();
                Rectangle oldRect = computeCurrentImageRect(imageDrawArea);
                updateZoomAnchorIfNeeded(e.getPoint(), oldRect);
                double nextZoom = e.getWheelRotation() < 0 ? zoom * ZOOM_STEP : zoom / ZOOM_STEP;
                zoom = clampZoom(nextZoom, imageDrawArea);
                positionImagePointAtScreenPoint(zoomAnchorImageX, zoomAnchorImageY, zoomAnchorScreen);
                repaint();
            }

            @Override
            public void mouseMoved(MouseEvent e) {
                if (zoomAnchorScreen != null && !zoomAnchorScreen.equals(e.getPoint())) {
                    clearZoomAnchor();
                }
            }
        };
        addMouseListener(mouseAdapter);
        addMouseMotionListener(mouseAdapter);
        addMouseWheelListener(mouseAdapter);
    }

    public void setDrawEnabled(boolean drawEnabled) {
        this.drawEnabled = drawEnabled;
        clearZoomAnchor();
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
        cancelRenderWorker();
        this.previewSource = null;
        clearRenderedViewport();
        this.title = null;
        this.zoom = 1.0;
        this.expandedToFill = false;
        this.panX = 0.0;
        this.panY = 0.0;
        this.dragStartScreen = null;
        this.panStartScreen = null;
        clearZoomAnchor();
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
        cancelRenderWorker();
        this.previewSource = rai == null ? null : new RaiPreviewSource<T>(rai);
        clearRenderedViewport();
        this.sourceVersion++;
        this.title = title;
        this.zoom = 1.0;
        this.expandedToFill = false;
        this.panX = 0.0;
        this.panY = 0.0;
        this.panStartScreen = null;
        clearZoomAnchor();
        this.boxColor = BOX_COLOR;
        this.titleColor = HELP_TEXT;
        clearBoxes();
        updateExpandButtonState();
        showDefaultHint();
        repaint();
    }

    public void setBufferedImage(BufferedImage image, String title) {
        setBufferedImage(image, title, true);
    }

    public void setBufferedImage(BufferedImage image, String title, boolean showHint) {
        cancelRenderWorker();
        this.previewSource = image == null ? null : new BufferedImagePreviewSource(image);
        clearRenderedViewport();
        this.sourceVersion++;
        this.title = title;
        this.zoom = 1.0;
        this.expandedToFill = false;
        this.drawEnabled = false;
        this.panX = 0.0;
        this.panY = 0.0;
        this.dragStartScreen = null;
        this.panStartScreen = null;
        clearZoomAnchor();
        this.activeBox = null;
        this.boxes.clear();
        this.boxColor = BOX_COLOR;
        this.titleColor = HELP_TEXT;
        updateToolTip();
        updateExpandButtonState();
        if (image != null && showHint) {
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

    public void setTitleColor(Color color) {
        this.titleColor = color == null ? HELP_TEXT : color;
        repaint();
    }

    public void setTitle(String title) {
        this.title = title;
        repaint();
    }

    public void setExpandedToFill(boolean expandedToFill) {
        this.expandedToFill = expandedToFill;
        this.zoom = 1.0;
        this.panX = 0.0;
        this.panY = 0.0;
        clearZoomAnchor();
        cancelRenderWorker();
        clearRenderedViewport();
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

        if (previewSource == null) {
            g2.setColor(EMPTY_BG);
            g2.fillRect(0, 0, getWidth(), getHeight());
            g2.setColor(HELP_TEXT);
            drawCenteredMessage(g2, emptyMessage);
            g2.dispose();
            return;
        }

        imageDrawArea = computeImageDrawArea();
        currentImageRect = computeCurrentImageRect(imageDrawArea);
        RenderGeometry renderGeometry = computeRenderGeometry(imageDrawArea, currentImageRect);

        g2.setClip(imageDrawArea);
        g2.setColor(EMPTY_BG);
        g2.fillRect(imageDrawArea.x, imageDrawArea.y, imageDrawArea.width, imageDrawArea.height);
        if (renderGeometry != null) {
            requestRender(renderGeometry.request);
            if (renderGeometry.request.equals(renderedRequest) && renderedViewport != null) {
                Rectangle screen = renderGeometry.screenRect;
                g2.drawImage(renderedViewport, screen.x, screen.y, screen.width, screen.height, null);
            } else {
                g2.setColor(HELP_TEXT);
                drawCenteredMessage(g2, "Rendering preview...");
            }
        }
        drawBoxes(g2, currentImageRect.x, currentImageRect.y, currentImageRect.width, currentImageRect.height);
        g2.setClip(null);

        if (title != null && !title.isEmpty()) {
            g2.setColor(titleColor);
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
        if (hintAlpha <= 0.0f || previewSource == null) {
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
        double sx = drawW / (double) previewSource.width();
        double sy = drawH / (double) previewSource.height();

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
        double imgRatio = previewSource.width() / (double) previewSource.height();
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

    private void updateCurrentImageRect() {
        if (previewSource == null) {
            currentImageRect = new Rectangle();
            return;
        }
        imageDrawArea = computeImageDrawArea();
        currentImageRect = computeCurrentImageRect(imageDrawArea);
    }

    private RenderGeometry computeRenderGeometry(Rectangle drawArea, Rectangle imageRect) {
        if (previewSource == null || imageRect.width <= 0 || imageRect.height <= 0) {
            return null;
        }
        Rectangle visibleScreen = imageRect.intersection(drawArea);
        if (visibleScreen.width <= 0 || visibleScreen.height <= 0) {
            return null;
        }

        double sourceX = (visibleScreen.x - imageRect.x) * previewSource.width() / (double) imageRect.width;
        double sourceY = (visibleScreen.y - imageRect.y) * previewSource.height() / (double) imageRect.height;
        double sourceW = visibleScreen.width * previewSource.width() / (double) imageRect.width;
        double sourceH = visibleScreen.height * previewSource.height() / (double) imageRect.height;
        Rectangle2D.Double sourceRect = clampSourceRect(sourceX, sourceY, sourceW, sourceH);
        if (sourceRect.width <= 0.0 || sourceRect.height <= 0.0) {
            return null;
        }

        int targetW = Math.max(1, Math.min(MAX_RENDER_SIDE, visibleScreen.width));
        int targetH = Math.max(1, Math.min(MAX_RENDER_SIDE, visibleScreen.height));
        double pixelCount = targetW * (double) targetH;
        if (pixelCount > MAX_RENDER_PIXELS) {
            double scale = Math.sqrt(MAX_RENDER_PIXELS / pixelCount);
            targetW = Math.max(1, (int) Math.floor(targetW * scale));
            targetH = Math.max(1, (int) Math.floor(targetH * scale));
        }

        RenderRequest request = new RenderRequest(sourceVersion, sourceRect, targetW, targetH);
        return new RenderGeometry(visibleScreen, request);
    }

    private Rectangle2D.Double clampSourceRect(double x, double y, double width, double height) {
        double x1 = Math.max(0.0, Math.min(previewSource.width(), x));
        double y1 = Math.max(0.0, Math.min(previewSource.height(), y));
        double x2 = Math.max(0.0, Math.min(previewSource.width(), x + width));
        double y2 = Math.max(0.0, Math.min(previewSource.height(), y + height));
        return new Rectangle2D.Double(x1, y1, Math.max(0.0, x2 - x1), Math.max(0.0, y2 - y1));
    }

    private void positionImagePointAtScreenPoint(double imageX, double imageY, Point screenPoint) {
        Rectangle area = computeImageDrawArea();
        Rectangle baseRect = computeBaseImageRect(area);
        int drawnW = Math.max(1, (int) Math.round(baseRect.width * zoom));
        int drawnH = Math.max(1, (int) Math.round(baseRect.height * zoom));
        int centeredX = area.x + (area.width - drawnW) / 2;
        int centeredY = area.y + (area.height - drawnH) / 2;

        double targetX = screenPoint.x - imageX * drawnW / (double) previewSource.width();
        double targetY = screenPoint.y - imageY * drawnH / (double) previewSource.height();
        panX = targetX - centeredX;
        panY = targetY - centeredY;
        currentImageRect = computeCurrentImageRect(area);
    }

    private double clampZoom(double requestedZoom, Rectangle area) {
        if (previewSource == null) {
            return 1.0;
        }
        Rectangle baseRect = computeBaseImageRect(area);
        double minVisibleByWidth = previewSource.width() * area.width
                / (double) Math.max(1, baseRect.width) / MIN_VISIBLE_SOURCE_SIZE;
        double minVisibleByHeight = previewSource.height() * area.height
                / (double) Math.max(1, baseRect.height) / MIN_VISIBLE_SOURCE_SIZE;
        double maxZoom = Math.max(1.0, Math.min(minVisibleByWidth, minVisibleByHeight));
        return Math.max(MIN_ZOOM, Math.min(maxZoom, requestedZoom));
    }

    private void updateZoomAnchorIfNeeded(Point screenPoint, Rectangle imageRect) {
        if (zoomAnchorScreen != null && zoomAnchorScreen.equals(screenPoint)) {
            return;
        }
        zoomAnchorScreen = new Point(screenPoint);
        zoomAnchorImageX = previewSource.width() / 2.0;
        zoomAnchorImageY = previewSource.height() / 2.0;
        if (imageRect.width > 0 && imageRect.height > 0 && imageRect.contains(screenPoint)) {
            zoomAnchorImageX = (screenPoint.x - imageRect.x) * previewSource.width() / (double) imageRect.width;
            zoomAnchorImageY = (screenPoint.y - imageRect.y) * previewSource.height() / (double) imageRect.height;
        }
    }

    private void clearZoomAnchor() {
        zoomAnchorScreen = null;
    }

    private void clearRenderedViewport() {
        renderedViewport = null;
        renderedRequest = null;
        pendingRequest = null;
    }

    private void cancelRenderWorker() {
        renderVersion++;
        if (renderWorker != null && !renderWorker.isDone()) {
            renderWorker.cancel(true);
        }
        renderWorker = null;
    }

    private void requestRender(RenderRequest request) {
        if (request == null || previewSource == null) {
            return;
        }
        if (request.equals(renderedRequest) || request.equals(pendingRequest)) {
            return;
        }
        cancelRenderWorker();
        pendingRequest = request;
        final long workerVersion = renderVersion;
        final PreviewSource source = previewSource;
        final RenderRequest workerRequest = request;
        renderWorker = new SwingWorker<RenderResult, Void>() {
            @Override
            protected RenderResult doInBackground() {
                BufferedImage image = source.render(workerRequest.sourceRect, workerRequest.targetW,
                        workerRequest.targetH, () -> isCancelled() || workerVersion != renderVersion);
                return image == null ? null : new RenderResult(workerRequest, image);
            }

            @Override
            protected void done() {
                if (isCancelled() || workerVersion != renderVersion) {
                    return;
                }
                try {
                    RenderResult result = get();
                    if (result == null || !result.request.equals(pendingRequest)) {
                        return;
                    }
                    renderedViewport = result.image;
                    renderedRequest = result.request;
                    pendingRequest = null;
                    repaint();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (CancellationException | ExecutionException e) {
                    if (workerVersion == renderVersion) {
                        pendingRequest = null;
                    }
                }
            }
        };
        renderWorker.execute();
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
                relX1 * previewSource.width(),
                relY1 * previewSource.height(),
                Math.max(0, (relX2 - relX1) * previewSource.width()),
                Math.max(0, (relY2 - relY1) * previewSource.height()));
    }

    private static int scaleToByte(double value, double min, double max) {
        double norm = (value - min) / (max - min);
        norm = Math.max(0, Math.min(1, norm));
        return (int) Math.round(norm * 255.0);
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private interface RenderAbortCheck {
        boolean shouldAbort();
    }

    private interface PreviewSource {
        int width();

        int height();

        BufferedImage render(Rectangle2D.Double sourceRect, int targetW, int targetH, RenderAbortCheck abortCheck);
    }

    private static final class BufferedImagePreviewSource implements PreviewSource {
        private final BufferedImage image;

        private BufferedImagePreviewSource(BufferedImage image) {
            this.image = image;
        }

        @Override
        public int width() {
            return image.getWidth();
        }

        @Override
        public int height() {
            return image.getHeight();
        }

        @Override
        public BufferedImage render(Rectangle2D.Double sourceRect, int targetW, int targetH,
                RenderAbortCheck abortCheck) {
            if (abortCheck.shouldAbort()) {
                return null;
            }
            BufferedImage out = new BufferedImage(targetW, targetH, BufferedImage.TYPE_INT_RGB);
            Graphics2D g2 = out.createGraphics();
            g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            int sx1 = clamp((int) Math.floor(sourceRect.x), 0, image.getWidth() - 1);
            int sy1 = clamp((int) Math.floor(sourceRect.y), 0, image.getHeight() - 1);
            int sx2 = clamp((int) Math.ceil(sourceRect.x + sourceRect.width), sx1 + 1, image.getWidth());
            int sy2 = clamp((int) Math.ceil(sourceRect.y + sourceRect.height), sy1 + 1, image.getHeight());
            g2.drawImage(image, 0, 0, targetW, targetH, sx1, sy1, sx2, sy2, null);
            g2.dispose();
            return out;
        }
    }

    private static final class RaiPreviewSource<T extends RealType<T> & NativeType<T>> implements PreviewSource {
        private final RandomAccessibleInterval<T> rai;
        private final long minX;
        private final long minY;
        private final long minC;
        private final int width;
        private final int height;
        private final int channels;
        private double minValue = 0.0;
        private double maxValue = 1.0;
        private boolean rangeReady;

        private RaiPreviewSource(RandomAccessibleInterval<T> source) {
            RandomAccessibleInterval<T> view = source;
            while (view.numDimensions() > 3) {
                view = Views.hyperSlice(view, view.numDimensions() - 1, view.min(view.numDimensions() - 1));
            }
            if (view.numDimensions() == 3 && view.dimension(2) != 3 && view.dimension(2) != 4) {
                view = Views.hyperSlice(view, 2, view.min(2));
            }
            this.rai = view;
            this.minX = view.min(0);
            this.minY = view.min(1);
            this.minC = view.numDimensions() > 2 ? view.min(2) : 0L;
            this.width = Math.max(1, (int) Math.min(Integer.MAX_VALUE, view.dimension(0)));
            this.height = Math.max(1, (int) Math.min(Integer.MAX_VALUE, view.dimension(1)));
            this.channels = view.numDimensions() > 2 ? Math.max(1, (int) Math.min(4, view.dimension(2))) : 1;
        }

        @Override
        public int width() {
            return width;
        }

        @Override
        public int height() {
            return height;
        }

        @Override
        public BufferedImage render(Rectangle2D.Double sourceRect, int targetW, int targetH,
                RenderAbortCheck abortCheck) {
            if (!ensureRange(abortCheck)) {
                return null;
            }
            BufferedImage out = new BufferedImage(targetW, targetH, BufferedImage.TYPE_INT_RGB);
            RandomAccess<T> access = rai.randomAccess();
            for (int y = 0; y < targetH; y++) {
                if (abortCheck.shouldAbort()) {
                    return null;
                }
                long sourceY = minY + sourceCoordinate(sourceRect.y, sourceRect.height, y, targetH, height);
                for (int x = 0; x < targetW; x++) {
                    long sourceX = minX + sourceCoordinate(sourceRect.x, sourceRect.width, x, targetW, width);
                    access.setPosition(sourceX, 0);
                    access.setPosition(sourceY, 1);
                    if (channels == 1) {
                        int v = scaleToByte(access.get().getRealDouble(), minValue, maxValue);
                        out.setRGB(x, y, (v << 16) | (v << 8) | v);
                    } else {
                        int[] rgb = new int[] {0, 0, 0};
                        for (int c = 0; c < Math.min(3, channels); c++) {
                            access.setPosition(minC + c, 2);
                            rgb[c] = scaleToByte(access.get().getRealDouble(), minValue, maxValue);
                        }
                        out.setRGB(x, y, (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]);
                    }
                }
            }
            return out;
        }

        private boolean ensureRange(RenderAbortCheck abortCheck) {
            if (rangeReady) {
                return true;
            }
            double[] range = estimateRange(rai, abortCheck);
            if (range == null) {
                return false;
            }
            minValue = range[0];
            maxValue = range[1] <= range[0] ? range[0] + 1.0 : range[1];
            rangeReady = true;
            return true;
        }

        private static long sourceCoordinate(double origin, double size, int targetIndex, int targetSize,
                int sourceLimit) {
            double source = origin + (targetIndex + 0.5) * size / Math.max(1, targetSize);
            return Math.max(0L, Math.min(sourceLimit - 1L, (long) Math.floor(source)));
        }

        private static <T extends RealType<T> & NativeType<T>> double[] estimateRange(
                RandomAccessibleInterval<T> rai, RenderAbortCheck abortCheck) {
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            int width = Math.max(1, (int) Math.min(Integer.MAX_VALUE, rai.dimension(0)));
            int height = Math.max(1, (int) Math.min(Integer.MAX_VALUE, rai.dimension(1)));
            int channels = rai.numDimensions() > 2 ? Math.max(1, (int) Math.min(3, rai.dimension(2))) : 1;
            int step = Math.max(1, (int) Math.ceil(Math.sqrt(width * (double) height / RANGE_SAMPLE_BUDGET)));
            RandomAccess<T> access = rai.randomAccess();
            for (int y = 0; y < height; y += step) {
                if (abortCheck.shouldAbort()) {
                    return null;
                }
                for (int x = 0; x < width; x += step) {
                    access.setPosition(rai.min(0) + x, 0);
                    access.setPosition(rai.min(1) + y, 1);
                    for (int c = 0; c < channels; c++) {
                        if (rai.numDimensions() > 2) {
                            access.setPosition(rai.min(2) + c, 2);
                        }
                        double value = access.get().getRealDouble();
                        min = Math.min(min, value);
                        max = Math.max(max, value);
                    }
                }
            }
            if (!Double.isFinite(min) || !Double.isFinite(max)) {
                return new double[] {0.0, 1.0};
            }
            return new double[] {min, max};
        }
    }

    private static final class RenderGeometry {
        private final Rectangle screenRect;
        private final RenderRequest request;

        private RenderGeometry(Rectangle screenRect, RenderRequest request) {
            this.screenRect = screenRect;
            this.request = request;
        }
    }

    private static final class RenderRequest {
        private final long sourceVersion;
        private final Rectangle2D.Double sourceRect;
        private final int targetW;
        private final int targetH;

        private RenderRequest(long sourceVersion, Rectangle2D.Double sourceRect, int targetW, int targetH) {
            this.sourceVersion = sourceVersion;
            this.sourceRect = sourceRect;
            this.targetW = targetW;
            this.targetH = targetH;
        }

        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof RenderRequest)) {
                return false;
            }
            RenderRequest other = (RenderRequest) obj;
            return sourceVersion == other.sourceVersion
                    && targetW == other.targetW
                    && targetH == other.targetH
                    && Double.compare(sourceRect.x, other.sourceRect.x) == 0
                    && Double.compare(sourceRect.y, other.sourceRect.y) == 0
                    && Double.compare(sourceRect.width, other.sourceRect.width) == 0
                    && Double.compare(sourceRect.height, other.sourceRect.height) == 0;
        }

        @Override
        public int hashCode() {
            int result = Long.hashCode(sourceVersion);
            long bits = Double.doubleToLongBits(sourceRect.x);
            result = 31 * result + Long.hashCode(bits);
            bits = Double.doubleToLongBits(sourceRect.y);
            result = 31 * result + Long.hashCode(bits);
            bits = Double.doubleToLongBits(sourceRect.width);
            result = 31 * result + Long.hashCode(bits);
            bits = Double.doubleToLongBits(sourceRect.height);
            result = 31 * result + Long.hashCode(bits);
            result = 31 * result + targetW;
            result = 31 * result + targetH;
            return result;
        }
    }

    private static final class RenderResult {
        private final RenderRequest request;
        private final BufferedImage image;

        private RenderResult(RenderRequest request, BufferedImage image) {
            this.request = request;
            this.image = image;
        }
    }
}
