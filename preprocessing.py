import numpy as np
from drawing import Drawing


class Transformer:
    def fit(self, X):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# Skalira crteže u [0, 1]x[0, 1], tako da budu što veći:
# - veću dimenziju skalira u [0, 1]
# - manju dimenziju centrira unutar [0, 1]
class DrawingScaler(Transformer):
    def fit(self, drawings: list[Drawing]):
        pass

    def transform(self, drawings: list[Drawing]):
        aligned_drawings = []
        for drawing in drawings:
            x_min = min(stroke[:, 0].min() for stroke in drawing.strokes)
            x_max = max(stroke[:, 0].max() for stroke in drawing.strokes)
            y_min = min(stroke[:, 1].min() for stroke in drawing.strokes)
            y_max = max(stroke[:, 1].max() for stroke in drawing.strokes)
            width = x_max - x_min
            height = y_max - y_min
            assert height > 0 or width > 0
            if width > height:
                scale = 1 / width
                shift = np.array([-x_min * scale, -scale * 0.5 * (y_min + y_max) + 0.5])
            else:
                scale = 1 / height
                shift = np.array([-scale * 0.5 * (x_min + x_max) + 0.5, -y_min * scale])
            aligned_drawings.append(Drawing(
                label=drawing.label,
                strokes=[scale * stroke.astype(np.float32) + shift for stroke in drawing.strokes],
            ))
        return aligned_drawings


class VideoTransformer(Transformer):
    def __init__(self, height, width, pen_width, n_frames):
        self.height = height
        self.width = width
        self.pen_width = pen_width
        self.n_frames = n_frames

    def fit(self, drawings: list[Drawing]):
        pass

    def transform(self, drawings: list[Drawing]) -> np.ndarray:
        pass
