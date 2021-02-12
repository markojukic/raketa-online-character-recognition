import numpy as np
from drawing import Drawing
from math import ceil
from typing import Any


class DrawingTransformer:
    def transform_one(self, drawing: Drawing) -> Any:
        raise NotImplementedError()

    def transform(self, drawings: list[Drawing]):
        return [self.transform_one(drawing) for drawing in drawings]


# Skalira crteže u [a, b]x[c, d], tako da budu što veći
class DrawingToBoxScaler(DrawingTransformer):
    def __init__(self, a: float, b: float, c: float, d: float, dtype=np.float32):
        assert a < b and c < d
        self.x_min = a
        self.y_min = c
        self.width = b - a
        self.height = d - c
        self.ratio = (d - c) / (b - a)
        self.x_mid = 0.5 * (a + b)
        self.y_mid = 0.5 * (c + d)
        self.dtype = dtype

    def transform_one(self, drawing: Drawing) -> Drawing:
        x_min = min(stroke[:, 0].min() for stroke in drawing.strokes)
        x_max = max(stroke[:, 0].max() for stroke in drawing.strokes)
        y_min = min(stroke[:, 1].min() for stroke in drawing.strokes)
        y_max = max(stroke[:, 1].max() for stroke in drawing.strokes)
        width = x_max - x_min
        height = y_max - y_min
        assert height > 0 or width > 0
        ratio = height / width
        if ratio < self.ratio:
            scale = self.width / width
            shift = np.array([
                self.x_min - x_min * scale,
                self.y_mid - scale * 0.5 * (y_min + y_max),
            ], dtype=self.dtype)
        else:
            scale = self.height / height
            shift = np.array([
                self.x_mid - scale * 0.5 * (x_min + x_max),
                self.y_min - y_min * scale,
            ], dtype=self.dtype)
        return Drawing(
            label=drawing.label,
            strokes=[scale * stroke.astype(self.dtype) + shift for stroke in drawing.strokes],
        )


def remove_duplicate_points(drawing: Drawing) -> Drawing:
    strokes = []
    for stroke in drawing.strokes:
        # Uklanjamo točke koje se ponavljaju više puta zaredom
        unique_mask = np.empty_like(stroke, dtype=np.bool)
        unique_mask[0] = True
        unique_mask[1:] = (stroke[1:] != stroke[:-1])
        unique_mask = np.logical_or.reduce(unique_mask, 1)
        stroke = stroke[unique_mask]
        strokes.append(stroke)
    return Drawing(label=drawing.label, strokes=strokes)


def stroke_times(drawing: Drawing) -> list[np.ndarray]:
    times = []
    for stroke in drawing.strokes:
        # Računamo vrijeme (duljinu puta) od prve do svake točke
        d = np.diff(stroke, axis=0)
        time = np.empty(stroke.shape[0])
        time[0] = 0
        time[1:] = np.hypot(d[:, 0], d[:, 1]).cumsum()
        times.append(time)
    return times


class DrawingResampler(DrawingTransformer):
    def __init__(self, n: int):
        self.n = n

    def transform_one(self, drawing: Drawing) -> Drawing:
        n = self.n
        drawing = remove_duplicate_points(drawing)
        times = stroke_times(drawing)

        n_strokes = len(drawing.strokes)
        stroke_lengths = np.array([time[-1] for time in times])
        stroke_lengths_order = np.argsort([time[-1] for time in times])
        stroke_lengths_sorted = stroke_lengths[stroke_lengths_order]
        remaining_lengths = np.cumsum(stroke_lengths_sorted[::-1])[::-1]
        resampled_strokes = [None] * n_strokes

        # Prvo radimo resampling za kraće strokove, da bismo rezervirali dovoljan broj točki za svaki
        for i in range(n_strokes):
            # Resampling za i-ti najkraći stroke
            order = stroke_lengths_order[i]
            stroke = drawing.strokes[order]
            time = times[order]
            stroke_length = stroke_lengths_sorted[i]
            k = ceil(n * stroke_length / remaining_lengths[i])
            if len(stroke) >= 1:
                k = max(k, 1)
            if len(stroke) >= 2:
                k = max(k, 2)

            resampled_time = np.linspace(0, stroke_length, k)
            resampled_strokes[order] = np.column_stack((
                np.interp(resampled_time, time, stroke[:, 0]).astype(np.float32),
                np.interp(resampled_time, time, stroke[:, 1]).astype(np.float32),
            ))

            n -= k
            assert n >= 0  # Ako znakovi nemaju previše strokova, za dovoljno veliki n bi trebalo raditi

        return Drawing(label=drawing.label, strokes=resampled_strokes)


class VideoCreator(DrawingTransformer):
    def __init__(self, n: int):
        self.n = n

    def transform_one(self, drawing: Drawing) -> list[Drawing]:
        video = []
        drawing = remove_duplicate_points(drawing)
        times = stroke_times(drawing)
        stroke_cum_times = np.array([0] + [time[-1] for time in times]).cumsum()
        i = 1
        for frame in range(1, self.n + 1):
            # Ukupna duljina strokova u frame-u
            last_point_time = (frame / self.n) * stroke_cum_times[-1]  # > 0
            while True:
                if last_point_time <= stroke_cum_times[i]:
                    break
                i += 1
            # Duljine zadnjeg stroka
            last_point_time -= stroke_cum_times[i - 1]
            time = times[i - 1]
            stroke = drawing.strokes[i - 1]
            time_until_last_point = np.append(time[time < last_point_time], last_point_time)
            last_stroke = np.column_stack((
                np.interp(time_until_last_point, time, stroke[:, 0]).astype(np.float32),
                np.interp(time_until_last_point, time, stroke[:, 1]).astype(np.float32),
            ))
            video.append(Drawing(label=drawing.label, strokes=drawing.strokes[:i - 1] + [last_stroke]))
        return video
