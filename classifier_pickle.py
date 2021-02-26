import numpy as np
import preprocessing
from drawing import Drawing
from dataclasses import dataclass
import dtw
import pickle
from typing import Any


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


class ClassifierPickle:
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


@dataclass
class DTWClassifierPickle(ClassifierPickle):
    stacked_train: np.ndarray
    drawing_scaler: preprocessing.DrawingToBoxScaler
    drawing_resampler: preprocessing.DrawingResampler
    cls: Any  # Sklearn classifier

    def predict(self, drawing: Drawing):
        drawing = self.drawing_scaler.transform_one(drawing)
        drawing = self.drawing_resampler.transform_one(drawing)
        d = dtw.dtw_distance_matrix(np.vstack(drawing.strokes)[None, :], self.stacked_train)
        return self.cls.predict(d)[0]


@dataclass
class RSIDTWClassifierPickle(DTWClassifierPickle):
    n_iter: int

    def predict(self, drawing: Drawing):
        drawing = self.drawing_scaler.transform_one(drawing)
        drawing = self.drawing_resampler.transform_one(drawing)
        d = dtw.rsi_dtw_distance_matrix(np.vstack(drawing.strokes)[None, :], self.stacked_train, self.n_iter)
        return self.cls.predict(d)[0]
