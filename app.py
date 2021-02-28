import sys
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import numpy as np
from typing import Optional
from drawing import Drawing
from classifier_pickle import load, DTWClassifierPickle, RSIDTWClassifierPickle
import matplotlib.pyplot as plt

class Buttons(QWidget):
    def __init__(self, parent: QWidget, h: int) -> None:
        super().__init__(parent)
        self.dtwCls: DTWClassifierPickle = load("models/KNN-DTW.pickle")
        self.rsidtwCls: RSIDTWClassifierPickle = load("models/KNN-RSIDTW.pickle")
        self.setFixedHeight(h)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addStretch(0)
        self.button1 = QPushButton("Obriši")
        self.button2 = QPushButton("Predvidi")
        self.button1.setFixedHeight(h/3)
        self.button2.setFixedHeight(h/3)
        self.button1.clicked.connect(parent.canvas.clearImage)
        self.button2.clicked.connect(lambda : self.predict(parent.canvas.drawing))
        self.layout.addWidget(self.button1, alignment=Qt.AlignTop)
        self.layout.addWidget(self.button2, alignment=Qt.AlignTop)

    def plot_drawing(self, drawing: Drawing):
        for stroke in drawing.strokes:
            plt.plot(stroke[:, 0], stroke[:, 1], color='black', linewidth=1, solid_capstyle='round')
        plt.axis('scaled')
        plt.show()

    def predict(self, drawing: Drawing):
        self.plot_drawing(drawing)
        text = "DTW: {0}\nRSIDTW: {1}".format(self.dtwCls.predict(drawing), self.rsidtwCls.predict(drawing))
        self.parent().notepad.setPlainText(text)

class Canvas(QWidget):
    def __init__(self, parent: QWidget, h: int) -> None:
        super().__init__(parent)
        self.h = h
        self.setFixedHeight(h)
        self.drawing: Drawing = Drawing("", [])
        self.curStroke = []
        self.image = QImage()
        self.penDown: bool = False
        self.lastPoint: QPoint = QPoint()
        self.penWidth = 4
        self.setAttribute(Qt.WA_StaticContents)
        self.brush = QBrush(Qt.GlobalColor.black)
        self.pen = QPen(self.brush, self.penWidth, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.dtype = np.float32

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
        self.drawing = Drawing("", [])
        self.curStroke = []
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
            self.curStroke.append((self.lastPoint.x(), -1 * self.lastPoint.y()))

    def mouseMoveEvent(self, event: QMouseEvent):
        if (event.buttons() & Qt.LeftButton) and self.penDown:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.penDown:
            self.drawLineTo(event.pos())
            self.penDown = False
            self.drawing.strokes.append(np.array(self.curStroke))
            print(self.drawing.strokes)
            self.curStroke = []

    def drawLineTo(self, endPoint: QPoint):
        painter = QPainter(self.image)
        painter.setPen(self.pen)
        painter.drawLine(self.lastPoint, endPoint)
        r = self.penWidth // 2 + 2
        self.update(QRect(self.lastPoint, endPoint).normalized().adjusted(-r, -r, r, r))
        self.lastPoint = endPoint
        self.curStroke.append((self.lastPoint.x(), -1 * self.lastPoint.y()))

class Notepad(QTextEdit):
    def __init__(self, parent: QWidget, h: int) -> None:
        super().__init__(parent)
        self.setFixedHeight(h)
        self.setFontPointSize(16)
        self.setPlainText("HELLO WORLD!")
        print(self.toPlainText())

class MainWindow(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("RaketaPad")
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(QMargins(0, 0, 0, 0))
        self.layout.setSpacing(0)
        self.canvas = Canvas(self, 200)
        self.notepad = Notepad(self, 100)
        self.buttons = Buttons(self, 100)

        self.layout.addWidget(self.buttons)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.notepad)
        self.setLayout(self.layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = MainWindow()
    mainWindow.resize(800, 500)
    mainWindow.show()

    sys.exit(app.exec_())
