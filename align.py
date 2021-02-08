import numpy as np
from drawing import Drawing


class DrawingAligner:
    def fit(self, drawings: list[Drawing]):
        pass

    def transform(self, drawings: list[Drawing]):
        aligned_drawings = []
        for drawing in drawings:
            x_min = min(stroke[:, 0].min() for stroke in drawing.strokes)
            x_max = max(stroke[:, 0].max() for stroke in drawing.strokes)
            y_min = min(stroke[:, 1].min() for stroke in drawing.strokes)
            y_max = max(stroke[:, 1].max() for stroke in drawing.strokes)
            assert x_min < x_max
            assert y_min < y_max
            aligned_drawing = Drawing(label=drawing.label, strokes=[])
            for stroke in drawing.strokes:
                aligned_stroke = stroke.astype(np.float32)
                # Horizontalno centriranje: x_min + x_max -> 0
                aligned_stroke[:, 0] -= (x_min + x_max) / 2
                # Skaliranje: y_min -> 0, y_max -> 1
                aligned_stroke[:, 1] -= y_min
                aligned_stroke /= (y_max - y_min)
                aligned_drawing.strokes.append(aligned_stroke)
            aligned_drawings.append(aligned_drawing)
        return aligned_drawings

    def fit_transform(self, drawings: list[Drawing]):
        self.fit(drawings)
        return self.transform(drawings)
