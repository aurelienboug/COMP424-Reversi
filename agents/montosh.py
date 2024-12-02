# Student agent: Add your own agent here
import copy

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

TIME_LIMIT = 1.93  # Don't exceed two seconds

position_scores_6x6 = [
    [100, -20, 10, 10, -20, 100],
    [-20, -50, 5, 5, -50, -20],
    [10, 5, 1, 1, 5, 10],
    [10, 5, 1, 1, 5, 10],
    [-20, -50, 5, 5, -50, -20],
    [100, -20, 10, 10, -20, 100],
]

position_scores_8x8 = [
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10, -2, 1, 1, 1, 1, -2, 10],
    [5, -2, 1, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 1, -2, 5],
    [10, -2, 1, 1, 1, 1, -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100],
]

position_scores_10x10 = [
    [100, -20, 10, 5, 5, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -2, -2, -50, -20],
    [10, -2, 1, 1, 1, 1, 1, 1, -2, 10],
    [5, -2, 1, 0, 0, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 0, 0, 1, -2, 5],
    [10, -2, 1, 1, 1, 1, 1, 1, -2, 10],
    [-20, -50, -2, -2, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 5, 5, 10, -20, 100],
]

position_scores_12x12 = [
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
    [100, -20, 10, 5, 5, 5, 5, 5, 5, 10, -20, 100],
]

weight_matrices = {6: position_scores_6x6, 8: position_scores_8x8, 10: position_scores_10x10,
                   12: position_scores_12x12}


@register_agent("montosh")
class Montosh(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(Montosh, self).__init__()
        self.start_time = None
        self.name = "Montosh"
        self.depth_limit = 3

    def step(self, chess_board, player, opponent):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (board_size, board_size)
          where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
          and 2 represents Player 2's discs (Brown).
        - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
        - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

        You should return a tuple (r,c), where (r,c) is the position where your agent
        wants to place the next disc. Use functions in helpers to determine valid moves
        and more helpful tools.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        self.start_time = time.time()
        best_move = None

        # Iterative deepening
        while not self.time_limit():
            move = self.alpha_beta_search(chess_board, player, opponent)

            # Only update the best move if we completed the whole depth
            if not self.time_limit():
                best_move = move

            self.depth_limit += 1

        # time_taken = time.time() - self.start_time

        # print("My AI's turn took ", time_taken, "seconds.")

        self.depth_limit = 3  # Reset depth limit
        if (best_move is not None):
            return best_move
        else:
            return random_move(chess_board, player)

    # Check if time limit has been exceeded for iterative deepening
    def time_limit(self):
        return time.time() - self.start_time >= TIME_LIMIT

    def alpha_beta_search(self, chess_board, player, opponent):
        best_move = None
        best_val = float("-inf")

        moves = get_valid_moves(chess_board, player)

        for valid_move in moves:

            new_state = deepcopy(chess_board)
            execute_move(new_state, valid_move, player)

            value = self.min_value(new_state, best_val, float('inf'), player, opponent, depth=1)

            if value > best_val:
                best_val = value
                best_move = valid_move

        return best_move

    # Max value function for alpha-beta pruning
    def max_value(self, chess_board, alpha, beta, player, opponent, depth):

        # If game over

        if depth > self.depth_limit or check_endgame(chess_board, player, opponent)[0] \
                or self.time_limit():
            score = evaluate(chess_board, player, opponent)
            return score

        value = float("-inf")

        moves = get_valid_moves(chess_board, player)
        if not moves:  # No valid moves, skip the turn
            return self.min_value(chess_board, alpha, beta, player, opponent, depth + 1)

        # Sort moves
        sorted_moves = sort_moves(moves, chess_board)

        # Random shuffle
        # np.random.shuffle(moves)

        for valid_move in sorted_moves:

            new_state = deepcopy(chess_board)
            execute_move(new_state, valid_move, player)

            value = max(value, self.min_value(new_state, alpha, beta, player, opponent, depth + 1))

            if value >= beta:
                return value

            alpha = max(alpha, value)

        return value

    # Min value function for alpha-beta pruning
    def min_value(self, chess_board, alpha, beta, player, opponent, depth):

        # If game over

        if depth > self.depth_limit or check_endgame(chess_board, player, opponent)[0] \
                or self.time_limit():
            score = evaluate(chess_board, player, opponent)
            return score

        value = float("inf")

        moves = get_valid_moves(chess_board, opponent)
        if not moves:  # No valid moves, skip the turn
            return self.max_value(chess_board, alpha, beta, player, opponent, depth + 1)

        # Sort moves
        sorted_moves = sort_moves(moves, chess_board)

        # Random shuffle
        # np.random.shuffle(moves)

        for valid_move in sorted_moves:

            new_state = deepcopy(chess_board)
            execute_move(new_state, valid_move, opponent)

            value = min(value, self.max_value(new_state, alpha, beta, player, opponent, depth + 1))

            if value <= alpha:
                return value

            beta = min(beta, value)

        return value


# current_player should be the player calling the sort (max or min)
def sort_moves(moves, chess_board):
    eval_moves = []
    for move in moves:
        # Append tuple of move and score.
        eval_moves.append((move, sort_heuristic(move, chess_board.shape[0])))

    # Sort based on eval score
    eval_moves.sort(key=lambda t: t[1], reverse=True)

    # Return a list of the moves only
    return [move for move, score in eval_moves]


# Scoring moves with a simple, low-cost function. Assumes a square board of size 6,8,10,12
def sort_heuristic(move, board_size):
    weight_matrix = weight_matrices[board_size]

    r, c = move
    score = weight_matrix[r][c]

    return score


# Eval function for alpha-beta pruning
def evaluate(chess_board, player, opponent):
    # difference in scores

    score_diff = np.sum(chess_board == player) - np.sum(chess_board == opponent)

    # HEURISTICS FROM GPT AGENT
    # Corner positions are highly valuable
    corners = [(0, 0), (0, chess_board.shape[1] - 1), (chess_board.shape[0] - 1, 0),
               (chess_board.shape[0] - 1, chess_board.shape[1] - 1)]
    corner_score = sum(1 for corner in corners if chess_board[corner] == player) * 10
    corner_penalty = sum(1 for corner in corners if chess_board[corner] == opponent) * -10

    # Mobility: the number of moves the opponent can make
    opponent_moves = len(get_valid_moves(chess_board, opponent))
    mobility_score = -opponent_moves

    return score_diff + corner_score + corner_penalty + mobility_score

