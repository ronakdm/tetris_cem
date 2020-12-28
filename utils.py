"""
    general utility functions
"""

import numpy as np
from joblib import Parallel, delayed

# make these imports last
import sys

sys.path.append("./envs")
from tetris import TetrisEnv, TetrisState

from features import DellacherieFeatureMap


class Player:
    def __init__(self, weights):
        self.env = TetrisEnv()
        self.env.reset()  # need this to init the state

        self.weights = weights  # TODO: init with Dellacherie weights
        self.feature_map = DellacherieFeatureMap()

        self.prev_lines_cleared = 0

    def step(self):
        # action = (orient, slot)
        a = self.action()

        _, _, done, _ = self.env.step(a)

        # Weighted sum of reward functions
        # I don't think we need this since for evaluating the weights all
        # we care about is lines cleared

        return done

    def action(self):
        best_val = -float("inf")
        best_a = None

        env = TetrisEnv()
        # TODO: parallelize with joblib or np.fromfunc
        for a in self.env.legal_moves[self.env.state.next_piece]:

            # Note: I think we probably don't need to create a new TetrisEnv each
            # iteration. we can just save self.env.state and keep resetting
            # the state of self.env to that state
            env.set_state(self.env.state)
            env.step(a)

            # we have self.env.state.next_piece, we have a(orient, slot),
            # we have state, env.cleared_current_turn

            features = self.feature_map.map(self.env, env, self.env.state.next_piece, a)
            val = np.dot(self.weights, features)

            best_a = a if val > best_val else best_a
            best_val = max(best_val, val)

        return best_a

    def get_total_lines_cleared(self):
        """
            Returns the number of rows cleared so far
        """
        return self.env.state.cleared


def eval_weights(num_episodes, weights, max_turns=float('inf')):

    # all_lines_cleared = np.empty(num_episodes, dtype=int)

    def worker():
        p = Player(weights)

        # run agent until end of game
        i = 0
        while not p.step() and i <= max_turns:
            i += 1

        return p.get_total_lines_cleared()

    all_lines_cleared = np.array(Parallel(n_jobs=-2)(
        delayed(worker)() for i in range(num_episodes)
    ))

    # return all_lines_cleared (by all players as a list)
    # TODO: Ronak: I suggest mean because I think we are going for the average performance
    # of the player over episodes?
    return all_lines_cleared
    # return np.round(np.mean(all_lines_cleared), decimals=2)
