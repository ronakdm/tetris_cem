import numpy as np
import sys

sys.path.append("./envs")
import tetris

from tetris import TetrisState
from features import DellacherieFeatureMap
from numpy.testing import assert_almost_equal

feature_map = DellacherieFeatureMap()


def test1():

    # This is the example used in "How to design good Tetris players".
    # We don't actually know what pieces were played, so we will use use 1.
    field = np.array(
        [
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            [0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    top = np.array([7, 10, 11, 11, 10, 9, 11, 12, 11, 14])
    next_piece = 0
    lost = False
    turn = 0
    cleared = 0
    state = TetrisState(field, top, next_piece, lost, turn, cleared)

    val = feature_map.f1(state)

    assert_almost_equal(val, 0, decimal=1)

    assert feature_map.f1(state) == 8, f"f1 was {feature_map.f1(state)}"
    assert feature_map.f2(state) == 4, f"f2 was {feature_map.f2(state)}"
    assert feature_map.f3(state) == 58, f"f3 was {feature_map.f3(state)}"
    assert feature_map.f4(state) == 45, f"f4 was {feature_map.f4(state)}"
    assert feature_map.f5(state) == 17, f"f5 was {feature_map.f5(state)}"
    assert feature_map.f6(state) == 8, f"f6 was {feature_map.f6(state)}"
    print("Test 1 passed")


if __name__ == "__main__":
    test1()
