# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves


@register_agent("min_opp_moves")
class MinOppAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(MinOppAgent, self).__init__()
        self.name = "MinOppAgent"

    def step(self, chess_board, player, opponent):
        start_time = time.time()

        valid_moves = get_valid_moves(chess_board, player)
        min = float('inf')
        min_move = valid_moves[0]
        for move in valid_moves:
            execute_move(chess_board,move, player)
            op_mv = get_valid_moves(chess_board, opponent)
            for opp in op_mv:
                cc = count_capture(chess_board, move,opp)
                if cc<min:
                    min = cc
                    min_move = move

        time_taken = time.time() - start_time

        print("My AI's turn took ", time_taken, "seconds.")
        return min_move
