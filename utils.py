"""
    General utility functions.
"""

import numpy as np
from joblib import Parallel, delayed

import sys

sys.path.append("./envs")
from tetris import TetrisEnv, TetrisState
from features import DellacherieFeatureMap


class Player:
    def __init__(self, weights):
        self.env = TetrisEnv()
        self.env.reset()  # need this to init the state

        self.weights = weights
        self.feature_map = DellacherieFeatureMap()

        self.prev_lines_cleared = 0

    def step(self):
        """ Take a step of the Tetris environment, and perform an action.

        Returns
        -------
        Boolean
            Whether the game has completed (the pieces have hit the ceiling).
        """
        # action = (orient, slot)
        a = self.action()

        _, _, done, _ = self.env.step(a)

        return done

    def action(self):
        """ Select a column and orientation for the new piece, given the current state of the board.

        Returns
        -------
        (orient, slot) : (int, int)
            orient is the piece orientation, int in 0:4, while slot is the column top which the
            piece belongs, int in 0:10.
        """
        best_val = -float("inf")
        best_a = None

        env = TetrisEnv()
        for a in self.env.legal_moves[self.env.state.next_piece]:

            env.set_state(self.env.state)
            env.step(a)

            features = self.feature_map.map(self.env, env, self.env.state.next_piece, a)
            val = np.dot(self.weights, features)

            best_a = a if val > best_val else best_a
            best_val = max(best_val, val)

        return best_a

    def get_total_lines_cleared(self):
        """
            Returns the number of rows cleared so far.
        """
        return self.env.state.cleared


def eval_weights(num_episodes, weights, max_turns=float("inf")):
    """[summary]

    Parameters
    ----------
    num_episodes : int
        For this set of weights, how many Tetris games should be played to evaluate reward.
    weights : numpy.ndarray
        Coefficients for each of the features of the state.
    max_turns : int, optional
        Max number of turns to evaluate each episode, by default float("inf")

    Returns
    -------
    all_lines_cleared : numpy.ndarray
        The number of lines cleared by these weights for each episode. (length num_episodes.)
    """

    def worker():
        p = Player(weights)

        # run agent until end of game
        i = 0
        while not p.step() and i <= max_turns:
            i += 1

        return p.get_total_lines_cleared()

    all_lines_cleared = np.array(
        Parallel(n_jobs=-2)(delayed(worker)() for i in range(num_episodes))
    )

    return all_lines_cleared
