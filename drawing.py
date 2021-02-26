from dataclasses import dataclass
import numpy as np


@dataclass
class Drawing:
    label: str
    strokes: list[np.array]