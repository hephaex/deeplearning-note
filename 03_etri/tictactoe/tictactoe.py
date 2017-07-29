# -*- coding: utf-8 -*-

import numpy as np


class TicTacToeEnv:

    def __init__(self):
        self.board = None
        self.current_player = None
        self.result = None
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=np.int32)
        self.current_player = "x"

    def step(self, index):
        if self.board[index] != 0:
            print("Invalid move!!")
            return None, None, None, {"valid": False}
        elif self.current_player == "x":
            self.board[index] = 1
            self.current_player = "o"
        else:
            self.board[index] = -1
            self.current_player = "x"
        observation = np.array(self.board)
        done, info = self.check_game_result()
        reward = 0
        return observation, reward, done, info

    def render(self):
        markers = []
        for i in self.board:
            if i == 0:
                markers.append("_")
            elif i == 1:
                markers.append("x")
            else:
                markers.append("o")
        print("{} is thinking...".format(self.current_player))
        print("{0}\t{1}\t{2}".format(markers[0], markers[1], markers[2]))
        print("{0}\t{1}\t{2}".format(markers[3], markers[4], markers[5]))
        print("{0}\t{1}\t{2}\n".format(markers[6], markers[7], markers[8]))

    def check_game_result(self):
        x_win, o_win, is_full = False, False, False
        # Check rows and cols
        for i in range(3):
            row = self.board[(i * 3):(i * 3 + 3)]
            col = self.board[i::3]
            if np.sum(row) == 3 or np.sum(col) == 3:
                x_win = True
            if np.sum(row) == -3 or np.sum(col) == -3:
                o_win = True
        # Check diag
        if np.sum(self.board[[0, 4, 8]]) == 3 or np.sum(self.board[[2, 4, 6]]) == 3:
            x_win = True
        if np.sum(self.board[[0, 4, 8]]) == -3 or np.sum(self.board[[2, 4, 6]]) == -3:
            o_win = True
        if 0 not in self.board:
            is_full = True
        done = x_win or o_win or is_full
        info = {"x": x_win, "o": o_win, "full": is_full, "valid": True}
        return done, info
