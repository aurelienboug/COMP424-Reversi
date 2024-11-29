# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

_board_weights =None

def generate_weights(board_size,player,board):

    weights = np.zeros((board_size, board_size))
    corner_value = 100
    near_corner_penalty = -70
    near_corner_diagonal_penalty = -50
    near_edge_value = 25
    edge_value = 50
    inner_value = 18 #(30*board_size)/ (board.size - np.count_nonzero(board))
    middle_four_value = 150/board_size

    middle_four = [(board_size//2 -1, board_size//2 -1),(board_size//2 -1, board_size//2),
                   (board_size//2, board_size//2 -1),(board_size//2, board_size//2)]
    corners = [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]
    near_corners = [(0, 1), (1, 0),(0, board_size - 2),(1, board_size - 1),(board_size - 2, 0),
                    (board_size - 1, 1) ,(board_size - 2, board_size - 1),(board_size - 1, board_size - 2)]
    near_corner_diagonal = [(1,1),(1,board_size-2),(board_size - 2, 1),(board_size - 2, board_size - 2)]

    for i in range(board_size):
        for j in range(board_size):
            if (i, j) in corners:
                weights[i, j] = corner_value
            elif (i,j) in middle_four:
                weights[i, j] = middle_four_value
            elif i in [0, board_size - 1] or j in [0, board_size - 1]:
                weights[i, j] = edge_value if (i, j) not in near_corners else near_corner_penalty
            elif (i,j) not in near_corner_diagonal and (i==board_size-2 or j==board_size-2 or i==1 or j==1):
                weights[i, j] = near_edge_value
            else:
                weights[i, j] = inner_value if (i, j) not in near_corner_diagonal else near_corner_diagonal_penalty


    for i in range(len(corners)):
        if board[corners[i]] == player:
            weights[near_corners[i*2]] *= -1
            weights[near_corners[i*2 +1]] *= -1
            weights[near_corner_diagonal[i][0]] *= -1

    return weights

def board_weights(board_size):
    global _board_weights
    if _board_weights is None or _board_weights.shape != (board_size, board_size):
        _board_weights = generate_weights(board_size)
        #print(_board_weights)
    return _board_weights

def heuristic(board, maximizing_player, player, opponent):
    """
    based on :
    1. positions captured on the board
    2.
    """
    board_size = board.shape[0]
    weights = generate_weights(board_size,player,board)
    #print(weights)
    #opponent_moves = get_valid_moves(board, player if maximizing_player else opponent) #####
    score = np.sum((board == player) * weights) - np.sum((board == opponent) * weights) #- opponent_moves
    return score


def alpha_beta(board, depth, alpha, beta, maximizing_player, player, opponent):
    """
    Perform Alpha-Beta pruning to find the best move.

    Parameters:
    - board: Current state of the board.
    - depth: Remaining depth to search.
    - alpha: Best value the maximizing player can guarantee.
    - beta: Best value the minimizing player can guarantee.
    - maximizing_player: True if the current player is maximizing, False otherwise.
    - player: The agent's player number.
    - opponent: The opponent's player number.

    Returns:
    - Best heuristic value and associated move.
    """

    valid_moves = get_valid_moves(board, player if maximizing_player else opponent)
    if depth == 0 or not valid_moves:  # or check_endgame(board,player,opponent):
        return heuristic(board, maximizing_player, player, opponent), None

    best_move = None
    if maximizing_player:
        max_eval = float('-inf')
        for move in valid_moves:
            new_board = deepcopy(board)
            execute_move(new_board, move, player)
            eval, _ = alpha_beta(new_board, depth - 1, alpha, beta, False, player, opponent)
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
            eval, _ = alpha_beta(new_board, depth - 1, alpha, beta, True, player, opponent)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval, best_move


@register_agent("comparor")
class Comparor(Agent):



    def __init__(self):
        super(Comparor, self).__init__()
        self.name = "Comparor"



    def step(self, chess_board, player, opponent):
        """
        Determine the best move using Alpha-Beta pruning and an advanced heuristic, adaptable for variable board sizes.

        Parameters:
        - chess_board: A numpy array representing the current state of the board.
        - player: The agent's player number (1 or 2).
        - opponent: The opponent's player number (1 or 2).

        Returns:
        - The optimal move (row, col) as a tuple.
        """

        # Adjust search depth based on board size
        board_size = chess_board.shape[0]
        depth =0
        start_time = time.time()
        while False:#(time.time()-start_time<2):
            search_depths = [0, 5, 4, 4, 3, 3, 2]
            depth = search_depths[board_size-6]  # Use deeper searches for smaller boards
            best_value = float('-inf')
            best_move = None
            value, move = alpha_beta(chess_board, depth, float('-inf'), float('inf'), True, player, opponent)
            if value > best_value:
                best_move = move
            depth += 1
        # Debugging: Time taken to calculate move
        value, best_move = alpha_beta(chess_board, 3, float('-inf'), float('inf'), True, player, opponent)
        time_taken = time.time() - start_time
        #print(f"Player {player}'s turn took {time_taken:.4f} seconds. For a board of size {board_size}.")
        #print("BEST MOVE :",best_move,"at depth ",depth," possible moves",len(get_valid_moves(chess_board, player)))
        return best_move
