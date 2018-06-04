"""
    Parentheses expressions generator.
"""
import numpy as np
from typing import Tuple

from constants import MAX_SEQ_LEN


def str_to_arr(seq: str) -> np.ndarray:
    arr = np.zeros([1, MAX_SEQ_LEN], dtype=np.float32)
    for i in range(len(seq)):
        c = seq[i]
        if c == '(':
            arr[0, i] = 1.
        elif c == ')':
            arr[0, i] = -1.
        else:
            raise ValueError("Invalid character encountered in bracket sequence")
    return arr


def compute_stats(seq: np.ndarray) -> np.ndarray:
    max_open = 0
    max_cons = 0
    max_dist = 0
    cur_cons = 0
    open_bracket_positions = []
    for pos in range(MAX_SEQ_LEN):
        if seq[0, pos] == 1.:
            cur_cons += 1
            open_bracket_positions.append(pos)
            max_open = max(max_open, len(open_bracket_positions))
            max_cons = max(max_cons, cur_cons)
        elif seq[0, pos] == -1.:
            cur_cons = 0
            assert open_bracket_positions, "invalid bracket sequence"
            max_dist = max(max_dist, pos-open_bracket_positions[-1])
            open_bracket_positions.pop()
        else:  # seq[0, pos] == 0.
            break
    assert not open_bracket_positions, "invalid bracket sequence"
    stats = np.array([max_open, max_cons, max_dist]).reshape([1, 3])
    return stats


def seq_len(seq: np.ndarray) -> int:
    return int(np.count_nonzero(seq))


class BrackSeqGen:
    def __init__(self):
        pass

    def generate_sequence_of_length(self, length=20) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class OpenCloseBrackSeqGen(BrackSeqGen):
    def __init__(self):
        super().__init__()

    def generate_sequence_of_length(self, length=20):
        seq = np.zeros([1, MAX_SEQ_LEN], dtype=np.float32)
        for i in range(length // 2):
            seq[0, i] = 1.
            seq[0, length // 2 + i] = -1.
        max_open = length // 2
        max_cons = length // 2
        max_dist = length - 1
        stats = np.array([max_open, max_cons, max_dist]).reshape([1, 3])
        return seq, stats

    def all_sequences(self):
        xs = []
        ys = []
        for length in range(MAX_SEQ_LEN):
            x, y = self.generate_sequence_of_length(length)
            xs.append(x)
            ys.append(y)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


class FixedParamsGen(BrackSeqGen):
    """ Generates bracket sequence for given params. """
    def __init__(self):
        super().__init__()

    @staticmethod
    def possible(length, max_cons, max_open, max_dist):
        return length % 2 == 0 and \
               max_dist % 2 == 1 and \
               max_open >= max_cons >= 2 and \
               length - 1 >= max_dist >= max_open * 2 + (max_open - max_cons) * 2

    def generate_sequence_of_length(self, length=20, max_cons=10, max_open=10, max_dist=19):
        if not self.possible(length, max_cons, max_open, max_dist):
            raise ValueError("Impossible to construct bracket sequence for such parameters")
        str_seq = ""
        # 1. Satisfy max_cons and max_open requirements.
        open = 0
        while True:
            cons = 0
            while cons < max_cons and open < max_open:
                str_seq += '('
                open += 1
                cons += 1
            str_seq += ')'
            if open == max_open:
                open -= 1
                break
            open -= 1
        # 2. Satisfy max_dist requirement.
        while len(str_seq) + open - 1 < max_dist:
            str_seq += '()'
        while open > 0:
            str_seq += ')'
            open -= 1
        # 3. Satisfy length requirement.
        while len(str_seq) < length:
            str_seq += '()'
        assert len(str_seq) == length
        seq = str_to_arr(str_seq)
        assert np.sum(seq) == 0.
        stats = np.array([max_open, max_cons, max_dist]).reshape([1, 3])
        assert np.array_equal(stats, compute_stats(seq))
        return seq, stats

    def all_sequences(self, max_length: int = MAX_SEQ_LEN):
        xs = []
        ys = []
        for length in range(2, max_length + 1, 2):
            for max_open in range(2, length // 2 + 1):
                for max_cons in range(2, max_open + 1):
                    for max_dist in range(max_open * 2 + (max_open - max_cons) * 2 + 1, length, 2):
                        if self.possible(length, max_cons, max_open, max_dist):
                            x, y = self.generate_sequence_of_length(length, max_cons, max_open, max_dist)
                            xs.append(x)
                            ys.append(y)
                        else:
                            assert False
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


class RandomBrackGen(BrackSeqGen):
    def __init__(self):
        super().__init__()

    def generate_sequence_of_length(self, length=20):
        assert length % 2 == 0
        str_seq = ''
        while len(str_seq) < length:
            idx = np.random.randint(0, len(str_seq) + 1)
            str_seq = str_seq[:idx] + '()' + str_seq[idx:]
        seq = str_to_arr(str_seq)
        stats = compute_stats(seq)
        return seq, stats
