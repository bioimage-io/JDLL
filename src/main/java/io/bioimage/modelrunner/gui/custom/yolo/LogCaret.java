package io.bioimage.modelrunner.gui.custom.yolo;

import java.awt.Rectangle;
import java.util.function.BooleanSupplier;

import javax.swing.text.DefaultCaret;
import javax.swing.text.Position;

/** Keep Swing's deferred caret repaint from scrolling after a programmatic tail update. */
final class LogCaret extends DefaultCaret {
    private static final long serialVersionUID = 1L;
    private final BooleanSupplier automaticMove;
    private boolean lastMoveAutomatic;

    LogCaret(BooleanSupplier automaticMove) {
        this.automaticMove = automaticMove;
        setUpdatePolicy(NEVER_UPDATE);
    }

    @Override public void setDot(int dot, Position.Bias bias) {
        lastMoveAutomatic = automaticMove.getAsBoolean();
        super.setDot(dot, bias);
    }

    @Override public void moveDot(int dot, Position.Bias bias) {
        lastMoveAutomatic = automaticMove.getAsBoolean();
        super.moveDot(dot, bias);
    }

    @Override protected void adjustVisibility(Rectangle area) {
        if (!lastMoveAutomatic) super.adjustVisibility(area);
    }
}
