from pathlib import Path
from typing import Optional, Tuple
from drawing import Drawing
from dataclasses import dataclass
import numpy as np


@dataclass
class UJIDrawing(Drawing):
    writer_id: str


# Parser za UJI Pen Characters dataset
class Parser:
    def __init__(self, directory):
        self.data_path = Path(directory).joinpath('ujipenchars2.txt')

    def __get_non_comment_line(self, file):
        for line in file:
            if line.startswith('//'):
                continue
            return line

    def parse(self, characters: Optional[str] = None) -> Tuple[list[UJIDrawing], list[UJIDrawing]]:
        train_entries: list[UJIDrawing] = []
        test_entries: list[UJIDrawing] = []
        with self.data_path.open('r', encoding='utf-8') as f:
            while True:
                line = self.__get_non_comment_line(f)
                if line is None:
                    break
                assert line.startswith('WORD ')
                _, label, session = line.split()
                writer_set, _, writer_id = session.split('_')
                line = self.__get_non_comment_line(f)
                assert line.strip().startswith('NUMSTROKES ')
                _, numstrokes = line.split()
                numstrokes = int(numstrokes)
                strokes = []
                for i in range(numstrokes):
                    line = self.__get_non_comment_line(f)
                    assert line.strip().startswith('POINTS')
                    _, points, _, stroke = line.split(maxsplit=3)
                    stroke = np.array(list(map(int, stroke.split()))).reshape((-1, 2))
                    stroke[:, 1] *= -1
                    strokes.append(stroke)
                if characters is None or label in characters:
                    drawing = UJIDrawing(label=label, strokes=strokes, writer_id=writer_id)
                    if writer_set == 'trn':
                        train_entries.append(drawing)
                    else:
                        test_entries.append(drawing)
        return train_entries, test_entries


if __name__ == '__main__':
    train, test = Parser('data/ujipenchars2').parse()
