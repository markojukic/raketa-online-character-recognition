import numpy as np
from drawing import Drawing
from math import ceil
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from typing import Any, Tuple
from multiprocessing import Pool


class DrawingTransformer:
    def transform_one(self, drawing: Drawing) -> Any:
        raise NotImplementedError()

    def transform(self, drawings: list[Drawing]):
        return [self.transform_one(drawing) for drawing in drawings]


def bounding_box(drawing: Drawing) -> Tuple[float, float, float, float]:
    return (
        min(stroke[:, 0].min() for stroke in drawing.strokes),
        max(stroke[:, 0].max() for stroke in drawing.strokes),
        min(stroke[:, 1].min() for stroke in drawing.strokes),
        max(stroke[:, 1].max() for stroke in drawing.strokes),
    )


def scale_shift_drawing(drawing: Drawing, scale: float, shift: np.ndarray, dtype=np.float32) -> Drawing:
    return Drawing(
        label=drawing.label,
        strokes=[(scale * stroke + shift).astype(dtype) for stroke in drawing.strokes],
    )


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
        x_min, x_max, y_min, y_max = bounding_box(drawing)
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
        return scale_shift_drawing(drawing, scale, shift, self.dtype)


# Skalira crteže tako da je ukupna duljina strokova 1 i centrira ih oko ishodišta
class DrawingLengthScaler(DrawingTransformer):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def transform_one(self, drawing: Drawing) -> Drawing:
        x_min, x_max, y_min, y_max = bounding_box(drawing)
        scale = 1 / sum(time[-1] for time in stroke_times(drawing))
        shift = -scale * 0.5 * np.array([x_min + x_max, y_min + y_max], dtype=self.dtype)
        return scale_shift_drawing(drawing, scale, shift, self.dtype)


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
            k = min(ceil(n * stroke_length / remaining_lengths[i]), n)
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


# Crte s okruglim rubovima
def rounded_line(draw, xy, width, fill):
    for i in range(len(xy)):
        draw.ellipse((
            xy[i, 0] - width + 1,
            xy[i, 1] - width + 1,
            xy[i, 0] + width - 1,
            xy[i, 1] + width - 1,
        ), fill=fill, outline=None)
    if xy.shape[0] > 1:
        draw.line(xy.flatten(), width=2 * width, fill=fill)


# Crta crtež kao sliku.
#   clip: dio ravnine koji se crta: (x_min, x_max, y_min, y_max)
#   size: širina i visina slike
#   width: debljina olovke
#   blur_radius: radius gaussian blura
#   preciznost: npr. za precision=2, sve se crta u 2x većoj rezoluciji, koja se na kraju smanjuje
class ImageDrawer:
    def __init__(self, clip=(0, 1, 0, 1), size=(32, 32), width=2, blur_radius=1, precision=2):
        self.output_size = size
        self.precision = precision
        self.image_size = (self.precision * size[0], self.precision * size[1])
        self.width = self.precision * width
        self.blur_radius = self.precision * blur_radius
        self.shift = np.array((clip[0], clip[2]), dtype=np.float32)
        self.scale = np.array((
            self.image_size[0] / (clip[1] - clip[0]),
            self.image_size[1] / (clip[3] - clip[2]),
        ), dtype=np.float32)

    @staticmethod
    def number_of_segments(drawing: Drawing) -> int:
        return sum(stroke.shape[0] for stroke in drawing.strokes) - 1

    # Crta sliku, svaki segment oboji odgovarajućom bojom
    def draw(self, drawing: Drawing, segment_colors: list[int] = None) -> np.ndarray:
        if segment_colors is None:
            return self.draw(drawing, [255] * ImageDrawer.number_of_segments(drawing))  # Sve crta crnom bojom
        assert ImageDrawer.number_of_segments(drawing) == len(segment_colors)

        image = Image.new('L', self.image_size)
        draw = ImageDraw.Draw(image)

        segment = 0
        for stroke in drawing.strokes:
            assert stroke.shape[0] > 1
            stroke = (stroke - self.shift) * self.scale
            for i in range(stroke.shape[0] - 1):
                rounded_line(draw, stroke[i:i + 1], self.width, segment_colors[segment])
                segment += 1
            segment += 1

        image = image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        image = image.resize(self.output_size)
        return np.flipud(np.asarray(image))


def create_gradient(values, n):
    values = np.array(values)
    gradient = np.interp(np.linspace(0, 1, n), values[:, 0], values[:, 1])
    return list(map(int, np.clip(gradient.astype(np.int64), 0, 255)))


# Crta crtež kao sliku, postepeno mijenja boju olovke. Za svaki gradijent crta po jednu sliku
class GradientCreator(DrawingTransformer):
    def __init__(self, gradients=None, **kwargs):
        self.image_drawer = ImageDrawer(**kwargs)
        if gradients is None:
            self.gradients = [
                list(range(1, 256)),  # bijela -> crna
                list(reversed(range(1, 256))),  # crna -> bijela
            ]
        else:
            self.gradients = gradients
        self.gradient_length = len(self.gradients[0])
        # Svi gradijenti jednake duljine
        assert all(len(g) == self.gradient_length for g in self.gradients)
        self.drawing_resampler = DrawingResampler(self.gradient_length + 1)  # Jednako segmenata kao boja

    def transform_one(self, drawing: Drawing) -> np.ndarray:
        drawing = self.drawing_resampler.transform_one(drawing)
        return np.stack([self.image_drawer.draw(drawing, gradient) for gradient in self.gradients])

    def transform(self, drawings: list[Drawing]):
        with Pool() as pool:
            return np.stack(pool.map(self.transform_one, drawings))


class VideoCreator(DrawingTransformer):
    def __init__(self, n: int, **kwargs):
        self.n = n
        self.image_drawer = ImageDrawer(**kwargs)

    def transform_one(self, drawing: Drawing) -> np.ndarray:
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
        return np.stack([self.image_drawer.draw(drawing) for drawing in video])

    def transform(self, drawings: list[Drawing]):
        with Pool() as pool:
            return np.stack(pool.map(self.transform_one, drawings))
