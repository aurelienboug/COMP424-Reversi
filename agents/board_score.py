# Agent that evaluates moves based on their position on the board

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

def board_weights(board_size):
    boards = [
        np.array([
    [100, -20, 10, 10, -20, 100],
    [-20, -50, 5, 5, -50, -20],
    [10, 5, 1, 1, 5, 10],
    [10, 5, 1, 1, 5, 10],
    [-20, -50, 5, 5, -50, -20],
    [100, -20, 10, 10, -20, 100]]),
        np.array([
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10, -2, 1, 1, 1, 1, -2, 10],
    [5, -2, 1, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 1, -2, 5],
    [10, -2, 1, 1, 1, 1, -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100]]),
        np.array([
    [100, -20, 10, 5, 5, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -2, -2, -50, -20],
    [10, -2, 1, 1, 1, 1, 1, 1, -2, 10],
    [5, -2, 1, 0, 0, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 0, 0, 1, -2, 5],
    [10, -2, 1, 1, 1, 1, 1, 1, -2, 10],
    [-20, -50, -2, -2, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 5, 5, 10, -20, 100]]),
        np.array([
    [100, -20, 10, 5, 5, 5, 5, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -2, -2, -2, -2, -50, -20],
    [10, -2, 1, 1, 1, 1, 1, 1, 1, 1, -2, 10],
    [5, -2, 1, 0, 0, 0, 0, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 0, 0, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 0, 0, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 0, 0, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 0, 0, 0, 0, 1, -2, 5],
    [10, -2, 1, 1, 1, 1, 1, 1, 1, 1, -2, 10],
    [-20, -50, -2, -2, -2, -2, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 5, 5, 5, 5, 10, -20, 100],
    [100, -20, 10, 5, 5, 5, 5, 5, 5, 10, -20, 100]])
    ]
    return boards[(board_size-6)//2]

def heuristic(board, player, opponent):
    weights = board_weights(board.shape[0])
    board_score = (np.sum((board == player) * weights) - np.sum((board == opponent) * weights))
    return board_score * 15

def alpha_beta(board, depth, alpha, beta, maximizing_player, player, opponent, start_time,move_order=None):

    valid_moves = get_valid_moves(board, player if maximizing_player else opponent)
    if depth == 0 or not valid_moves:
        return heuristic(board, player, opponent), None

    if time.time() - start_time >=2 :
        raise TimeoutError

    if move_order:
        valid_moves = sorted(valid_moves, key=lambda m: move_order.index(m) if m in move_order else len(move_order))

    best_move = None
    if maximizing_player:
        max_eval = float('-inf')
        for move in valid_moves:
            new_board = deepcopy(board)
            execute_move(new_board, move, player)
            eval, _ = alpha_beta(new_board, depth - 1, alpha, beta, False, player, opponent, start_time,move_order)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in valid_moves:
            new_board = deepcopy(board)
            execute_move(new_board, move, opponent)
            eval, _ = alpha_beta(new_board, depth - 1, alpha, beta, True, player,opponent, start_time,move_order)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval, best_move

@register_agent("board_score")
class Board_score(Agent):

    def __init__(self):
        super(Board_score, self).__init__()
        self.name = "board_score"

    def step(self, board, player, opponent):

        best_value = float('-inf')
        best_move = None
        depth = [5,4,4,3,3,2,2][board.shape[0]-6]
        move_order = []  # Store the best move from previous depths for ordering
        start_time = time.time()

        while True:

            if time.time() - start_time >= 2:
                break

            try:
                value, move = alpha_beta(board, depth, float('-inf'), float('inf'), True, player, opponent, start_time, move_order)
                if move is not None:
                    best_move = move
                    best_value = value
                    # Update move order to prioritize the current best move
                    move_order = [best_move] + [m for m in move_order if m != best_move]
                depth += 1

            except TimeoutError:
                break

        time_taken = time.time() - start_time
        return best_move
