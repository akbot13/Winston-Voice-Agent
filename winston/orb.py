"""Floating, always-on-top orb that shows what Winston is doing.

Click it to talk (same as the wake word), drag it anywhere, right-click for the menu.
"""
import json
import math
import os
import time

from PySide6.QtCore import QObject, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QApplication, QLabel, QMenu, QVBoxLayout, QWidget

STATES = {
    #            colour           pulse speed, pulse depth
    "loading":   ((120, 130, 150), 0.6, 0.04),
    "idle":      ((70, 140, 255),  0.35, 0.05),
    "listening": ((0, 220, 255),   1.2, 0.03),
    "thinking":  ((160, 90, 255),  0.8, 0.06),
    "speaking":  ((255, 185, 70),  2.2, 0.08),
    "asking":    ((255, 120, 40),  1.6, 0.10),
    "error":     ((255, 70, 70),   0.5, 0.04),
}
LABELS = {"loading": "Waking up…", "listening": "Listening…", "thinking": "Thinking…",
          "asking": "Waiting for your yes or no"}


class Bridge(QObject):
    """Thread-safe way for the Winston thread to update the orb."""
    state = Signal(str)
    caption = Signal(str, str)
    level = Signal(float)


class Caption(QWidget):
    """Speech-bubble under the orb with the latest line from you or Winston."""

    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
                         | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.label = QLabel(self)
        self.label.setWordWrap(True)
        self.label.setFont(QFont("Segoe UI", 10))
        self.label.setStyleSheet("color: #f2f4f8;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.addWidget(self.label)
        self.hide_timer = QTimer(self, singleShot=True, timeout=self.hide)

    def show_text(self, who, text, anchor, seconds=None):
        text = text if len(text) < 260 else text[:257] + "…"
        self.label.setText(f"<b>{who}</b>&nbsp; {text}" if who else text)
        self.setFixedWidth(300)
        self.adjustSize()
        screen = QApplication.screenAt(anchor) or QApplication.primaryScreen()
        area = screen.availableGeometry()
        x = min(max(anchor.x() - self.width() // 2, area.left() + 8), area.right() - self.width() - 8)
        y = anchor.y() + 6
        if y + self.height() > area.bottom():  # no room below the orb: put it above
            y = anchor.y() - 150 - self.height()
        self.move(x, y)
        self.show()
        self.hide_timer.start(int((seconds or max(4, len(text) / 14)) * 1000))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 14, 14)
        p.fillPath(path, QColor(18, 22, 34, 225))
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.drawPath(path)


class Orb(QWidget):
    SIZE = 140
    RADIUS = 30

    def __init__(self, bridge, state_file, on_click, on_new_chat, on_quit):
        super().__init__(None, Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.PointingHandCursor)
        self.on_click, self.on_new_chat, self.on_quit = on_click, on_new_chat, on_quit
        self.state_file = state_file
        self.captions_on = True
        self.caption = Caption()

        self.state = "loading"
        self.color = list(STATES["loading"][0])
        self.level_now = 0.0
        self.t0 = time.time()
        self.drag_from = None
        self.dragged = False

        bridge.state.connect(self.set_state)
        bridge.caption.connect(self.show_caption)
        bridge.level.connect(self.set_level)

        self._place()
        QTimer(self, interval=33, timeout=self.update).start()  # ~30 fps animation
        self.setToolTip("Winston: click to talk, drag to move, right-click for options")

    # --- placement -------------------------------------------------------------
    def _place(self):
        try:
            pos = json.load(open(self.state_file))
            self.move(pos["x"], pos["y"])
            if QApplication.screenAt(self.geometry().center()):
                return
        except Exception:
            pass
        area = QApplication.primaryScreen().availableGeometry()
        self.move(area.right() - self.SIZE - 24, area.bottom() - self.SIZE - 24)

    def _save_place(self):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            json.dump({"x": self.x(), "y": self.y()}, open(self.state_file, "w"))
        except OSError:
            pass

    def _anchor(self):
        return self.mapToGlobal(QPoint(self.SIZE // 2, self.SIZE // 2 + self.RADIUS + 22))

    # --- slots -----------------------------------------------------------------
    def set_state(self, state):
        if state not in STATES or state == self.state:
            return
        self.state = state
        if state in LABELS and self.captions_on:
            self.caption.show_text("", LABELS[state], self._anchor(), seconds=60 if state != "loading" else 120)
        elif state == "idle" and self.caption.label.text() in LABELS.values():
            self.caption.hide()

    def show_caption(self, who, text):
        if self.captions_on:
            self.caption.show_text(who, text, self._anchor())

    def set_level(self, level):
        self.level_now = max(self.level_now, min(level, 1.0))

    # --- mouse -----------------------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drag_from = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self.dragged = False

    def mouseMoveEvent(self, e):
        if self.drag_from is not None and e.buttons() & Qt.LeftButton:
            target = e.globalPosition().toPoint() - self.drag_from
            if (target - self.pos()).manhattanLength() > 3:
                self.dragged = True
            if self.dragged:
                self.move(target)
                self.caption.hide()

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        if self.dragged:
            self._save_place()
        else:
            self.on_click()
        self.drag_from = None

    def contextMenuEvent(self, e):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu{background:#141a28;color:#eef;border:1px solid #334;padding:4px}"
                           "QMenu::item{padding:6px 18px}QMenu::item:selected{background:#2a3a66}")
        for text, fn in (("Talk to Winston", self.on_click),
                         ("New conversation", self.on_new_chat),
                         ("Hide captions" if self.captions_on else "Show captions", self._toggle_captions),
                         ("Quit Winston", self.on_quit)):
            act = QAction(text, menu)
            act.triggered.connect(fn)
            menu.addAction(act)
        menu.exec(e.globalPos())

    def _toggle_captions(self):
        self.captions_on = not self.captions_on
        if not self.captions_on:
            self.caption.hide()

    # --- drawing ---------------------------------------------------------------
    def paintEvent(self, _):
        target, speed, depth = STATES[self.state]
        self.color = [c + (t - c) * 0.12 for c, t in zip(self.color, target)]  # ease between colours
        self.level_now *= 0.85
        t = time.time() - self.t0
        pulse = math.sin(t * speed * 2 * math.pi)
        if self.state == "asking":
            pulse = 1.0 if (t * 1.6) % 1 < 0.5 else -1.0
        scale = 1 + depth * pulse + 0.25 * self.level_now
        r = self.RADIUS * scale
        c = QPointF(self.SIZE / 2, self.SIZE / 2)
        red, green, blue = (int(v) for v in self.color)

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)

        # outer glow
        glow = QRadialGradient(c, r * 2.2)
        glow.setColorAt(0.0, QColor(red, green, blue, 150))
        glow.setColorAt(0.45, QColor(red, green, blue, 55))
        glow.setColorAt(1.0, QColor(red, green, blue, 0))
        p.setBrush(glow)
        p.drawEllipse(c, r * 2.2, r * 2.2)

        # body: lit from the top-left like a glass marble
        body = QRadialGradient(QPointF(c.x() - r * 0.35, c.y() - r * 0.4), r * 1.5)
        body.setColorAt(0.0, QColor(min(red + 120, 255), min(green + 120, 255), min(blue + 120, 255)))
        body.setColorAt(0.45, QColor(red, green, blue))
        body.setColorAt(1.0, QColor(int(red * 0.25), int(green * 0.25), int(blue * 0.3)))
        p.setBrush(body)
        p.drawEllipse(c, r, r)

        # thinking: two arcs orbiting the body
        if self.state == "thinking":
            p.setBrush(Qt.NoBrush)
            for i, alpha in ((0, 220), (180, 120)):
                p.setPen(QPen(QColor(255, 255, 255, alpha), 2.2, Qt.SolidLine, Qt.RoundCap))
                start = int((-t * 300 + i) * 16) % (360 * 16)
                p.drawArc(QRectF(c.x() - r - 7, c.y() - r - 7, 2 * r + 14, 2 * r + 14), start, 70 * 16)
            p.setPen(Qt.NoPen)

        # specular highlight
        shine = QRadialGradient(QPointF(c.x() - r * 0.38, c.y() - r * 0.45), r * 0.45)
        shine.setColorAt(0.0, QColor(255, 255, 255, 170))
        shine.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setBrush(shine)
        p.drawEllipse(QPointF(c.x() - r * 0.38, c.y() - r * 0.45), r * 0.45, r * 0.45)
