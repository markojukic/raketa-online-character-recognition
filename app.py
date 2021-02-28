import sys
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import numpy as np
from typing import Optional
from drawing import Drawing
from classifier_pickle import load
import time


class Canvas(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.drawing = Drawing('', [])
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
        newImage = QImage(newSize, QImage.Format_Grayscale8)
        newImage.fill(255)
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
        self.image.fill(255)
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        dirtyRect = event.rect()
        painter.drawImage(dirtyRect, self.image, dirtyRect)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.penDown = True
            self.lastPoint = event.pos()
            self.curStroke.append((self.lastPoint.x(), -self.lastPoint.y()))

    def mouseMoveEvent(self, event: QMouseEvent):
        if (event.buttons() & Qt.LeftButton) and self.penDown:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.penDown:
            self.drawLineTo(event.pos())
            self.penDown = False
            self.drawing.strokes.append(np.array(self.curStroke, dtype=np.float32))
            self.curStroke = []

    def drawLineTo(self, endPoint: QPoint):
        painter = QPainter(self.image)
        painter.setPen(self.pen)
        painter.drawLine(self.lastPoint, endPoint)
        r = self.penWidth // 2 + 2
        self.update(QRect(self.lastPoint, endPoint).normalized().adjusted(-r, -r, r, r))
        self.lastPoint = endPoint
        self.curStroke.append((self.lastPoint.x(), -self.lastPoint.y()))


class MainWindow(QDialog):
    def __init__(self, models: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Prava Raketa")
        self.models = models
        self.canvas = Canvas(self)
        self.erase_button = QPushButton("Obriši")
        self.predict_button = QPushButton("Predvidi")
        self.results = QTableWidget()
        self.results.setRowCount(len(self.models))
        self.results.setColumnCount(3)
        self.results.setHorizontalHeaderLabels(["Model", "Predikcija", "Vrijeme"])
        self.results.setFixedWidth(400)
        self.results.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results.verticalHeader().setVisible(False)
        self.results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i, name in enumerate(self.models):
            self.results.setItem(i, 0, QTableWidgetItem(name))
            self.results.setItem(i, 1, QTableWidgetItem())
            self.results.setItem(i, 2, QTableWidgetItem())

        # Events
        self.erase_button.clicked.connect(self.canvas.clearImage)
        self.predict_button.clicked.connect(self.print_predictions)

        # Fonts
        font = self.font()
        font.setPixelSize(18)
        self.erase_button.setFont(font)
        self.predict_button.setFont(font)
        font.setPixelSize(15)
        self.results.setFont(font)

        # Buttons
        self.erase_button.setFixedHeight(30)
        self.predict_button.setFixedHeight(30)

        # Layouts
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.erase_button)
        self.button_layout.addWidget(self.predict_button)

        self.results_layout = QVBoxLayout()
        self.results_layout.addLayout(self.button_layout)
        self.results_layout.addWidget(self.results)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.canvas)
        self.layout.addLayout(self.results_layout)
        self.setLayout(self.layout)

    def print_predictions(self):
        for i in range(len(self.models)):
            self.results.item(i, 1).setText('')
            self.results.item(i, 2).setText('')

        if self.canvas.drawing.strokes:
            for i, (_, cls) in enumerate(self.models.items()):
                start = time.time()
                self.results.item(i, 1).setText(cls.predict(self.canvas.drawing))
                self.results.item(i, 2).setText(f'{time.time() - start:.3f}s')


if __name__ == '__main__':
    models = {
        'KNN: DTW': load("models/KNN-DTW.pickle"),
        'KNN: RSIDTW': load("models/KNN-RSIDTW.pickle"),
    }

    app = QApplication(sys.argv)

    mainWindow = MainWindow(models)
    mainWindow.resize(900, 500)
    mainWindow.show()

    sys.exit(app.exec_())
