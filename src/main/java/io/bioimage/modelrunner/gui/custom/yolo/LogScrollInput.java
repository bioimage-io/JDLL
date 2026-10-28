package io.bioimage.modelrunner.gui.custom.yolo;

import java.awt.AWTEvent;
import java.awt.Component;
import java.awt.EventQueue;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;

import javax.swing.JScrollPane;
import javax.swing.SwingUtilities;

/** Layout and look-and-feel animations must not be mistaken for user scrolling. */
final class LogScrollInput {
    private LogScrollInput() {}

    static boolean isUserScroll(JScrollPane scroll) {
        if (scroll.getVerticalScrollBar().getValueIsAdjusting()) return true;
        AWTEvent event = EventQueue.getCurrentEvent();
        if (event == null || !(event.getSource() instanceof Component)) return false;
        if (!SwingUtilities.isDescendingFrom((Component) event.getSource(), scroll)) return false;
        if (event instanceof MouseEvent) return true;
        if (event instanceof KeyEvent) {
            switch (((KeyEvent) event).getKeyCode()) {
                case KeyEvent.VK_UP: case KeyEvent.VK_DOWN:
                case KeyEvent.VK_PAGE_UP: case KeyEvent.VK_PAGE_DOWN:
                case KeyEvent.VK_HOME: case KeyEvent.VK_END: return true;
                default: break;
            }
        }
        return false;
    }
}
