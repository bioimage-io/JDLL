package io.bioimage.modelrunner.gui.custom.yolo;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Component;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Insets;
import java.awt.RenderingHints;
import java.awt.geom.Path2D;
import java.util.List;
import java.util.function.IntConsumer;

import javax.swing.BorderFactory;
import javax.swing.ButtonModel;
import javax.swing.Icon;
import javax.swing.JButton;
import javax.swing.JPanel;

import io.bioimage.modelrunner.gui.custom.segmentation.SegmentationOutput;

/** A single action surface with independently operable previous, execute and next areas. */
final class SegmentationActionCarousel extends JPanel {
    private static final long serialVersionUID = 1L;
    private static final int BAR_WIDTH = 3, BAR_GAP = 3, INDICATOR_PAD = 4;
    private static final Color HOVER = new Color(213, 222, 236);
    private static final Color CHEVRON = new Color(80, 98, 120);
    private static final Color INACTIVE_BAR = new Color(161, 175, 193);
    private static final Color ACTIVE_BAR = new Color(82, 122, 165);

    private final JButton previous, run, next;
    private final List<SegmentationOutput> actions;
    private final int indicatorWidth;
    private int selection;

    SegmentationActionCarousel(JButton run, List<SegmentationOutput> actions, IntConsumer browse) {
        super(null);
        this.run = run;
        this.actions = actions;
        indicatorWidth = actions.size() * BAR_WIDTH + (actions.size() - 1) * BAR_GAP;
        setOpaque(true);
        setBackground(run.getBackground());
        setBorder(YoloUiUtils.BUTTON_BORDER);
        previous = arrow(-1, "Previous output", browse);
        next = arrow(1, "Next output", browse);
        addArea(previous);
        addArea(run);
        addArea(next);

        // Equal insets keep the label centered on the whole control, including the indicator.
        int reserve = indicatorWidth + 2 * INDICATOR_PAD;
        run.setBorder(BorderFactory.createEmptyBorder(0, reserve, 0, reserve));
    }

    private static JButton arrow(int direction, String name, IntConsumer browse) {
        Icon icon = new Chevron(direction);
        JButton button = new JButton(icon);
        button.setDisabledIcon(icon);
        button.setToolTipText(name);
        button.getAccessibleContext().setAccessibleName(name);
        button.addActionListener(e -> browse.accept(direction));
        return button;
    }

    private void addArea(JButton button) {
        button.setBorder(BorderFactory.createEmptyBorder());
        button.setContentAreaFilled(false);
        button.setOpaque(false);
        button.setFocusPainted(true);
        button.setRolloverEnabled(true);
        button.getModel().addChangeListener(e -> repaint(button.getBounds()));
        add(button);
    }

    void select(SegmentationOutput output, boolean canBrowse) {
        selection = actions.indexOf(output);
        previous.setEnabled(canBrowse);
        next.setEnabled(canBrowse);
        run.getAccessibleContext().setAccessibleName(output.actionLabel);
        run.getAccessibleContext().setAccessibleDescription("Action " + (selection + 1) + " of " + actions.size());
        repaint();
    }

    @Override public void doLayout() {
        Insets insets = getInsets();
        int width = Math.max(0, getWidth() - insets.left - insets.right);
        int height = Math.max(0, getHeight() - insets.top - insets.bottom);
        YoloUiUtils.applyResponsiveFont(run, getHeight());
        FontMetrics metrics = run.getFontMetrics(run.getFont());
        int longestCaption = 0;
        for (SegmentationOutput action : actions)
            longestCaption = Math.max(longestCaption, metrics.stringWidth(action.actionLabel));
        Insets padding = run.getInsets();
        int spaceForArrows = width - padding.left - padding.right - longestCaption;
        int arrowWidth = Math.min(Math.min(32, width / 7), Math.max(18, spaceForArrows / 2));
        previous.setBounds(insets.left, insets.top, arrowWidth, height);
        run.setBounds(insets.left + arrowWidth, insets.top, width - 2 * arrowWidth, height);
        next.setBounds(insets.left + width - arrowWidth, insets.top, arrowWidth, height);
        int textWidth = run.getWidth() - padding.left - padding.right;
        SegmentationOutput selected = actions.get(selection);
        String caption = selected == SegmentationOutput.BOTH && textWidth < longestCaption
                ? "Generate both" : selected.actionLabel;
        YoloUiUtils.setFullText(run, caption);
        YoloUiUtils.applyResponsiveText(run, textWidth, getHeight());
    }

    @Override protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        paintFeedback(g, previous);
        paintFeedback(g, run);
        paintFeedback(g, next);
        int x = run.getX() + run.getWidth() - indicatorWidth - INDICATOR_PAD;
        for (int i = 0; i < actions.size(); i++) {
            boolean active = i == selection;
            int h = Math.min(active ? 12 : 8, run.getHeight());
            g.setColor(active && previous.isEnabled() ? ACTIVE_BAR : INACTIVE_BAR);
            g.fillRect(x, run.getY() + (run.getHeight() - h) / 2, BAR_WIDTH, h);
            x += BAR_WIDTH + BAR_GAP;
        }
    }

    private static void paintFeedback(Graphics g, JButton button) {
        ButtonModel model = button.getModel();
        boolean pressed = model.isPressed() && model.isArmed();
        if (model.isEnabled() && (pressed || model.isRollover())) {
            g.setColor(pressed ? YoloUiUtils.INPUT_BORDER : HOVER);
            g.fillRect(button.getX(), button.getY(), button.getWidth(), button.getHeight());
        }
    }

    private static final class Chevron implements Icon {
        private final int direction;

        Chevron(int direction) { this.direction = direction; }
        @Override public int getIconWidth() { return 8; }
        @Override public int getIconHeight() { return 12; }

        @Override public void paintIcon(Component c, Graphics g, int x, int y) {
            Graphics2D pen = (Graphics2D) g.create();
            try {
                pen.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                pen.setStroke(new BasicStroke(1.35f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
                pen.setColor(c.isEnabled() ? CHEVRON : INACTIVE_BAR);
                Path2D path = new Path2D.Float();
                int mid = x + getIconWidth() / 2;
                path.moveTo(mid - 2 * direction, y + 2);
                path.lineTo(mid + 2 * direction, y + 6);
                path.lineTo(mid - 2 * direction, y + 10);
                pen.draw(path);
            } finally {
                pen.dispose();
            }
        }
    }
}
