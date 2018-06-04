"""
    Parentheses expressions generator.
"""
import numpy as np
from typing import Tuple

from constants import MAX_SEQ_LEN

Stats = Tuple[int, int, int]  # max_open, max_cons, max_dist


class BrackSeqGen:
    def __init__(self):
        pass

    def generate_sequence(self, length=20) -> Tuple[np.ndarray, Stats]:
        raise NotImplemented


class OpenCloseBrackSeqGen(BrackSeqGen):
    def __init__(self):
        super().__init__()

    def generate_sequence(self, length=20):
        seq = np.zeros([1, MAX_SEQ_LEN])
        for i in range(length // 2):
            seq[0, i] = 1.
            seq[0, length // 2 + i] = -1.
        max_open = length // 2
        max_cons = length // 2
        max_dist = length - 1
        return seq, (max_open, max_cons, max_dist)


if __name__ == '__main__':
    g = OpenCloseBrackSeqGen()
    print(g.generate_sequence(10))
