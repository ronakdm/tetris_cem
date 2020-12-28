import numpy as np

N_COLS = 10
N_ROWS = 21


class DellacherieFeatureMap:
    def __init__(self):
        pass

    def map(self, prev_env, env, piece_id, action):
        r"""
        Feature vector using Dellacherie's hand-designed features.
        Parameters
        ----------
        prev_env : TetrisEnv
        env : TetrisEnv
        piece : int
        action : (orient, slot)
        Returns
        -------
        value : ndarray
        """

        value = np.array(
            [
                self.f1(prev_env, action),
                self.f2(prev_env, env, piece_id, action),
                self.f3(env.state),
                self.f4(env.state),
                self.f5(env.state),
                self.f6(env.state),
            ]
        )

        return value

    def f1(self, prev_env, action):
        r"""
        Landing height: the height at which the current piece fell

        Parameters
        ----------
        prev_env : TetrisEnv
        env : TetrisEnv
        action : (orient, slot)
        Returns
        -------
        value : int
        """

        orient, slot = action

        return prev_env.state.top[slot]

    def f2(self, prev_env, env, piece_id, action):
        r"""
        Eroded pieces: the contribution of the last piece to the cleared lines
        times the number of cleared lines.

        Parameters
        ----------
        prev_env : TetrisEnv
        env : TetrisEnv
        piece_id : int
        action : (orient, slot)
        Returns
        -------
        value : int
        """

        orient, slot = action

        prev_col_height = prev_env.state.top[slot]
        curr_col_height = env.state.top[slot]
        piece_height = env.piece_height[piece_id][orient]

        contribution = prev_col_height + piece_height - curr_col_height

        return env.cleared_current_turn * contribution

    def f3(self, state):
        r"""
        Row transitions: number of filled cells adjacent to empty cells summed over all rows.

        Parameters
        ----------
        state : TetrisState
        Returns
        -------
        value : int
        """

        row_transitions = 0

        # Iterate over rows that have at least one cell filled
        for row in range(max(state.top)):
            # Iterate through row and count transitions
            #   prev and curr are 1 for filled and 0 for empty
            prev = 1
            for col in range(N_COLS):
                curr = 0 if (state.field[row][col] == 0) else 1
                if curr != prev:
                    row_transitions += 1
                prev = curr
            # empty cell next to right-side wall
            if prev == 0:
                row_transitions += 1

        return row_transitions

    def f4(self, state):
        r"""
        Column transition: same as (f3) summed over all columns;
        note that borders count as filled cells.

        Parameters
        ----------
        state : TetrisState
        Returns
        -------
        value : int
        """

        col_transitions = 0

        # Iterate over columns
        for col in range(N_COLS):
            prev = 1
            for row in range(state.top[col]):
                curr = 0 if (state.field[row][col]) else 1
                if curr != prev:
                    col_transitions += 1
                prev = curr

        return col_transitions

    def f5(self, state):
        r"""
        Number of holes: the number of empty cells with at least one filled cell above.

        Parameters
        ----------
        state : TetrisState
        Returns
        -------
        value : int
        """

        holes = 0

        # Loop through columns
        for col in range(N_COLS):
            # Loop through rows up till height of this column
            for row in range(state.top[col]):
                # If the cell above is empty, increment holes
                if (
                    (row < N_ROWS - 1)
                    and (state.field[row][col] == 0)
                    and (state.field[row + 1][col] != 0)
                ):
                    holes += 1

        return holes

    def f6(self, state):
        r"""
        Cumulative wells: the sum of the accumulated depths of the wells.

        Parameters
        ----------
        state : TetrisState
        Returns
        -------
        value : int
        """

        cum_well_depth = 0

        for i in range(len(state.top)):
            # Get  height of left column
            if i == 0:
                left_top = N_ROWS
            else:
                left_top = state.top[i - 1]

            # Get height of right column
            if i == N_COLS - 1:
                right_top = N_ROWS
            else:
                right_top = state.top[i + 1]

            # If both left and right heights are greater, then there is a well
            if (left_top > state.top[i]) and (right_top > state.top[i]):
                well_depth = min(left_top, right_top) - state.top[i]
                # Sum from 1 to well_depth
                cum_well_depth += sum(range(well_depth + 1))
                # cum_well_depth = (well_depth * (well_depth + 1)) // 2

        return cum_well_depth
