import torch
import random
import numpy as np
from collections import deque

from snake_game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:
    def __init__(self):
        self.number_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY)

    def get_state(self, game):
        pass

    def remember(self, state, action, reward, next_state, game_over):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, game_over):
        pass

    def get_action(self, state):
        pass


def train():
    plot_scores = []
    plot_avg_scores = []
    total_score = []
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        old_state = agent.get_state(game)

        # get move
        final_move = agent.get_action(old_state)

        # perform move and get a new state
        reward, game_over, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_long_memory(old_state, final_move,
                                reward, new_state, game_over)

        # remember
        agent.remember(old_state, final_move,
                       reward, new_state, game_over)

        if game_over:
            # train long memory and plot result
            game.reset()
            agent.number_games += 1
            agent.train_long_memory()

            if score > record:
                record = score

            print("Game:", agent.number_games)
            print("Score:", score)
            print("Record:", record)


if __name__ == '__main__':
    train()
