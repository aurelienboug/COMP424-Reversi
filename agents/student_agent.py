from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

TIME_LIMIT = 2


def ratio(num1, num2):
    result = 0
    if num1 + num2 != 0:
        result = (num1 - num2) / (num1 + num2)
    return result * 100

def mobility(board, player, opponent):
    return -len(get_valid_moves(board, opponent))

def parity(board, player, opponent):
    player_score = np.sum(board == player)
    opponent_score = np.sum(board == opponent)
    return ratio(player_score, opponent_score)

def corner_capture(board, player, opponent):
    player_corner_capture = 0
    opponent_corner_capture = 0
    board_size = board.shape[0]
    near_corner_player = 0
    near_corner_opponent = 0
    for corner in [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]:
        if board[corner] != 0:
            if board[corner] == player:
                player_corner_capture += 1
            else:
                opponent_corner_capture += 1
        else:
            for d in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                if 0 <= corner[0] + d[0] < board_size and 0 <= corner[1] + d[1] < board_size:
                    if board[(corner[0] + d[0], corner[1] + d[1])] == player:
                        near_corner_player += 1
                    if board[(corner[0] + d[0], corner[1] + d[1])] == opponent:
                        near_corner_opponent += 1

    return ratio(player_corner_capture, opponent_corner_capture), ratio(near_corner_opponent, near_corner_player)

def calculate_stability(board, player, opponent):
    """
    Calculates the stability levels of coins on the board for a given player.

    Stability levels:
        3: Super stable (corners)
        2: Lines and coins near corners
        1: Semi-stable (potentially stable but not guaranteed)
        0: Unstable (can be flipped)

    Args:
        board (np.ndarray): A 2D array representing the board state.
                            0 represents an empty cell, 1 represents the player's coin, and 2 the opponent's coin.
        player (int): The integer representing the player (1 or 2).
        opponent (int): The integer representing the opponent (1 or 2).

    Returns:
        np.ndarray: A 2D array where each cell contains the stability level of the player's coin:
                    3, 2, 1, or 0 for player coins, -3, -2, -1, or 0 for opponent coins.
    """
    board_size = board.shape[0]
    stability_board = np.zeros_like(board, dtype=int)  # Initialize stability board

    # Stability weights
    STABILITY_CORNER = 3
    STABILITY_LINE = 2
    STABILITY_SEMI = 1
    STABILITY_UNSTABLE = 0

    # Directions for neighbor checks
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    def is_corner(x, y):
        """Check if the position is a corner."""
        return (x, y) in [(0, 0), (0, board_size - 1),
                          (board_size - 1, 0), (board_size - 1, board_size - 1)]

    def is_near_corner(x, y):
        """Check if the position is near a corner."""
        return (x, y) in [(0, 1), (1, 0), (1, 1),
                          (0, board_size - 2), (1, board_size - 1), (1, board_size - 2),
                          (board_size - 2, 0), (board_size - 1, 1), (board_size - 2, 1),
                          (board_size - 2, board_size - 1), (board_size - 1, board_size - 2),
                          (board_size - 2, board_size - 2)]

    def is_line(x, y):
        """Check if the position is on the edge of the board (not a corner or near-corner)."""
        return (x in [0, board_size - 1] or y in [0, board_size - 1]) and not is_corner(x,
                                                                                        y) and not is_near_corner(x,
                                                                                                                  y)

    def is_semi_stable(x, y, player):
        """
        Check if a coin is semi-stable:
        It is surrounded by coins of the same type in some directions but not fully stable.
        """
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_size and 0 <= ny < board_size:
                if board[nx, ny] == 0 or board[nx, ny] == opponent:  # Empty or opponent coin nearby
                    return True
        return False

    # Assign stability levels to each cell
    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] == 0:  # Skip empty cells
                continue

            coin_owner = player if board[i, j] == player else opponent
            stability_level = STABILITY_UNSTABLE

            # Determine stability
            if is_corner(i, j):
                stability_level = STABILITY_CORNER
            elif is_near_corner(i, j) or is_line(i, j):
                stability_level = STABILITY_LINE
            elif is_semi_stable(i, j, coin_owner):
                stability_level = STABILITY_SEMI

            # Assign stability level to the appropriate owner
            stability_board[i, j] = stability_level if coin_owner == player else -stability_level

    return stability_board

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
    return boards[(board_size - 6) // 2]

def dynamic_weights_score(board, player, opponent):
    dw_board = board_weights(board.shape[0]) * calculate_stability(board, player, opponent)
    player_score = np.sum((board == player) * dw_board)
    opponent_score = np.sum((board == opponent) * dw_board)
    score = 0
    if (player_score != 0 and opponent_score != 0):
        score = (player_score + opponent_score) / (abs(player_score) + abs(opponent_score)) * 100
    return score

def start_game_heuristic(board, player, opponent):
    parity_score = parity(board, player, opponent)
    mobility_score = mobility(board, player, opponent)
    board_score = dynamic_weights_score(board, player, opponent)
    return mobility_score * 20 + board_score + parity_score

def mid_game_heuristic(board, player, opponent):
    (corners_score, near_corner_penalty) = corner_capture(board, player, opponent)
    board_score = dynamic_weights_score(board, player, opponent)
    mobility_score = mobility(board, player, opponent)
    return mobility_score * 20 + corners_score * 20 + board_score + near_corner_penalty  # *10

def end_game_heuristic(board, player, opponent):
    (corners_score, near_corner_penalty) = corner_capture(board, player, opponent)
    parity_score = parity(board, player, opponent)
    # board_score = dynamic_weights_score(board, player, opponent)
    return parity_score * 20 + corners_score * 10  # + near_corner_penalty +board_score

def heuristic(board, player, opponent):
    game_advancement = (np.sum((board == player)) + np.sum((board == opponent))) / board.shape[0] ** 2
    if game_advancement < 0.25:
        return start_game_heuristic(board, player, opponent)
    elif game_advancement < 0.8:
        return mid_game_heuristic(board, player, opponent)
    else:
        return end_game_heuristic(board, player, opponent)

def alpha_beta(board, depth, alpha, beta, maximizing_player, player, opponent, start_time, move_order=None):
    """
    Alpha-beta pruning with optional move ordering.

    Args:
        board (np.ndarray): The current state of the board.
        depth (int): Maximum search depth.
        alpha (float): Alpha value for pruning.
        beta (float): Beta value for pruning.
        maximizing_player (bool): Whether the current player is maximizing or minimizing.
        player (int): Maximizing player.
        opponent (int): Minimizing player.
        move_order (list): A list of moves ordered by priority from the previous iteration.

    Returns:
        tuple: The best evaluation value and the best move.
    """

    valid_moves = get_valid_moves(board, player if maximizing_player else opponent)
    if depth == 0 or not valid_moves:
        return heuristic(board, player, opponent), None

    if time.time() - start_time >= TIME_LIMIT:
        # print("alpha",time.time(),start_time)
        raise TimeoutError

    # Use move ordering if provided
    if move_order:
        valid_moves = sorted(valid_moves, key=lambda m: move_order.index(m) if m in move_order else len(move_order))

    best_move = None
    if maximizing_player:
        max_eval = float('-inf')
        for move in valid_moves:
            new_board = deepcopy(board)
            execute_move(new_board, move, player)
            eval, _ = alpha_beta(new_board, depth - 1, alpha, beta, False, player, opponent, start_time, move_order)
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
            eval, _ = alpha_beta(new_board, depth - 1, alpha, beta, True, player, opponent, start_time, move_order)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval, best_move


@register_agent("student_agent")
class StudentAgent(Agent):

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"

    def step(self, board, player, opponent):

        best_value = float('-inf')
        best_move = None
        depth = [5, 4, 4, 3, 3, 2, 2][board.shape[0] - 6]
        move_order = []  # Store the best move from previous depths for ordering
        start_time = time.time()
        # print("step",start_time)

        while True:

            if time.time() - start_time >= TIME_LIMIT:
                break

            try:
                # print(time.time() - start_time)
                # print(start_time)
                value, move = alpha_beta(board, depth, float('-inf'), float('inf'), True, player, opponent, start_time,
                                         move_order)
                if move is not None:
                    best_move = move
                    best_value = value
                    # Update move order to prioritize the current best move
                    move_order = [best_move] + [m for m in move_order if m != best_move]
                depth += 1

            except TimeoutError:
                break

        time_taken = time.time() - start_time

        """print(self.name)
        parity_score = parity(board, player, opponent)
        mobility_score = mobility(board, player, opponent)
        board_score = dynamic_weights_score(board, player, opponent)
        (corners_score, near_corner_penalty) = corner_capture(board, player, opponent)
        print("parity_score", parity_score, "board_score", board_score, "mobility_score", mobility_score,
              "corners_score", corners_score, "near_corner_penalty", near_corner_penalty)

        game_advancement = (np.sum((board == player)) + np.sum((board == opponent))) / board.shape[0] ** 2
        if game_advancement < 0.3:
            print("start heuristic: ",heuristic(board, player, opponent))
        elif game_advancement < 0.65:
            print("mid_game heuristic",heuristic(board, player, opponent))
        else:
            print("end_game heuristic",heuristic(board, player, opponent))


        #print(heuristic(board, player, opponent))
        #print(board)
        #print(f"Player {player}'s turn took {time_taken:.4f} seconds. For a board of size {board.shape[0]}.")
        print("BEST MOVE :",best_move,"at depth ",depth-1)

        print("------------------------------------------------------------------------------------------------------------")"""
        # 104 - 40
        return best_move
