import sys
from PySide6.QtGui import QResizeEvent, QTabletEvent
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from typing import Optional
import time


class Canvas(QWidget):
    def __init__(self, parent: QWidget, h: int) -> None:
        super().__init__(parent)
        self.h = h
        self.setFixedHeight(h)
        self.current_drawing: list[list[int]] = []
        self.image = QImage()
        self.penDown: bool = False
        self.lastPoint: QPoint = QPoint()
        self.penWidth = 4
        self.setAttribute(Qt.WA_StaticContents)
        self.brush = QBrush(Qt.GlobalColor.black)
        self.pen = QPen(self.brush, self.penWidth, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

    def resizeImage(self, image: QImage, newSize: QSize):
        if image.size() == newSize:
            return
        newImage = QImage(newSize, QImage.Format_RGB32)
        newImage.fill(qRgb(255, 255, 255))
        painter = QPainter(newImage)
        painter.drawImage(QPoint(0, 0), image)
        self.image = newImage

    def resizeEvent(self, event: QResizeEvent):
        if self.width() > self.image.width() or self.height() > self.image.height():
            newWidth = max(self.width() + 128, self.image.width())
            newHeight = max(self.height() + 128, self.image.height())
            self.resizeImage(self.image, QSize(newWidth, newHeight))
            self.update()
        super().resizeEvent(event)

    def clearImage(self) -> None:
        self.image.fill(qRgb(255, 255, 255))
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        dirtyRect = event.rect()
        painter.drawImage(dirtyRect, self.image, dirtyRect)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.penDown = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if (event.buttons() & Qt.LeftButton) and self.penDown:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.penDown:
            self.drawLineTo(event.pos())
            self.penDown = False

    def drawLineTo(self, endPoint: QPoint):
        painter = QPainter(self.image)
        painter.setPen(self.pen)
        painter.drawLine(self.lastPoint, endPoint)
        r = self.penWidth // 2 + 2
        self.update(QRect(self.lastPoint, endPoint).normalized().adjusted(-r, -r, r, r))
        self.lastPoint = endPoint


class Notepad(QTextEdit):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setFontPointSize(16)
        self.setPlainText("HELLO WORLD!")
        print(self.toPlainText())


class MainWindow(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("RaketaPad")

        self.VBoxLayout = QVBoxLayout()
        self.canvas = Canvas(self, 200)
        self.notepad = Notepad(self)

        self.VBoxLayout.addWidget(self.canvas)
        self.VBoxLayout.addWidget(self.notepad)
        self.setLayout(self.VBoxLayout)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = MainWindow()
    mainWindow.resize(800, 600)
    mainWindow.show()

    sys.exit(app.exec_())
