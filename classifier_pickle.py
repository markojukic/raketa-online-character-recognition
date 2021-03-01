import numpy as np
import preprocessing
from drawing import Drawing
from dataclasses import dataclass
import dtw
from joblib import dump
from typing import Any
from preprocessing import DrawingTransformer
from hosvdclassifier import HOSVDClassifier


class ClassifierPickle:
    def save(self, filename):
        dump(self, filename)


@dataclass
class KNNDTWPickle(ClassifierPickle):
    stacked_train: np.ndarray
    drawing_scaler: preprocessing.DrawingToBoxScaler
    drawing_resampler: preprocessing.DrawingResampler
    cls: Any  # Sklearn classifier

    def get_X(self, drawing: Drawing):
        return dtw.dtw_distance_matrix(np.vstack(drawing.strokes)[None, :], self.stacked_train)

    def predict(self, drawing: Drawing):
        drawing = self.drawing_scaler.transform_one(drawing)
        drawing = self.drawing_resampler.transform_one(drawing)
        return self.cls.predict(self.get_X(drawing))[0]


@dataclass
class KNNRSIDTWPickle(KNNDTWPickle):
    n_iter: int

    def get_X(self, drawing: Drawing):
        return dtw.rsi_dtw_distance_matrix(np.vstack(drawing.strokes)[None, :], self.stacked_train, self.n_iter)


@dataclass
class SVMDTWPickle(KNNDTWPickle):
    gamma: float

    def get_X(self, drawing: Drawing):
        return np.exp(-self.gamma * super().get_X(drawing))


@dataclass
class SVMRSIDTWPickle(KNNRSIDTWPickle):
    gamma: float

    def get_X(self, drawing: Drawing):
        return np.exp(-self.gamma * super().get_X(drawing))


@dataclass
class HOSVDPickle(ClassifierPickle):
    transformers: list[DrawingTransformer]
    cls: HOSVDClassifier

    def predict(self, drawing: Drawing):
        for transformer in self.transformers:
            drawing = transformer.transform_one(drawing)
        return self.cls.predict(drawing[None, :])[0]
