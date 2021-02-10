import numpy as np
from drawing import Drawing
from math import ceil


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
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

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
                shift = np.array([-x_min * scale, -scale * 0.5 * (y_min + y_max) + 0.5], dtype=self.dtype)
            else:
                scale = 1 / height
                shift = np.array([-scale * 0.5 * (x_min + x_max) + 0.5, -y_min * scale], dtype=self.dtype)
            aligned_drawings.append(Drawing(
                label=drawing.label,
                strokes=[scale * stroke.astype(self.dtype) + shift for stroke in drawing.strokes],
            ))
        return aligned_drawings


def resample_drawing(drawing: Drawing, n: int) -> Drawing:
    strokes = []
    times = []

    for stroke in drawing.strokes:
        # Uklanjamo točke koje se ponavljaju više puta zaredom
        unique_mask = np.empty_like(stroke, dtype=np.bool)
        unique_mask[0] = True
        unique_mask[1:] = (stroke[1:] != stroke[:-1])
        unique_mask = np.logical_or.reduce(unique_mask, 1)
        stroke = stroke[unique_mask]
        strokes.append(stroke)

        # Računamo vrijeme (duljinu puta) od prve do svake točke
        d = np.diff(stroke, axis=0)
        time = np.empty(stroke.shape[0])
        time[0] = 0
        time[1:] = np.hypot(d[:, 0], d[:, 1]).cumsum()
        times.append(time)

    n_strokes = len(strokes)
    stroke_lengths = np.array([time[-1] for time in times])
    stroke_lengths_order = np.argsort([time[-1] for time in times])
    stroke_lengths_sorted = stroke_lengths[stroke_lengths_order]
    remaining_lengths = np.cumsum(stroke_lengths_sorted[::-1])[::-1]
    resampled_strokes = [None] * n_strokes

    # Prvo radimo resampling za kraće strokove, da bismo rezervirali dovoljan broj točki za svaki
    for i in range(n_strokes):
        # Resampling za i-ti najkraći stroke
        order = stroke_lengths_order[i]
        stroke = strokes[order]
        time = times[order]
        stroke_length = stroke_lengths_sorted[i]
        k = ceil(n * stroke_length / remaining_lengths[i])
        if len(stroke) >= 1:
            k = max(k, 1)
        if len(stroke) >= 2:
            k = max(k, 2)

        resampled_time = np.linspace(0, stroke_length, k)
        resampled_strokes[order] = np.column_stack((
            np.interp(resampled_time, time, stroke[:, 0]),
            np.interp(resampled_time, time, stroke[:, 1]),
        ))

        n -= k
        assert n >= 0  # Ako znakovi nemaju previše strokova, za dovoljno veliki n bi trebalo raditi

    return Drawing(label=drawing.label, strokes=resampled_strokes)


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
