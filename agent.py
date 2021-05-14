import torch
import random
import numpy as np
from collections import deque

from snake_game import SnakeGameAI, Direction, Point
from model import DQN, Qtrain
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:
    def __init__(self):
        self.number_games = 0
        self.epsilon = 0

        # Discount rate
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        # State, hidden_layer, Action
        self.model = DQN(11, 256, 3)
        self.trainer = Qtrain(
            self.model, learn_rate=LEARNING_RATE, gamma=self.gamma
        )

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger forward
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        # popleft if we hit max memory
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # return list of tuples
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards,
                                next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        self.epsilon = 80 - self.number_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            prediction = self.model(
                torch.tensor(state, dtype=torch.float)
            )
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_avg_scores = []
    total_score = 0
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
        agent.train_short_memory(old_state, final_move,
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
                agent.model.save()

            print("Game:", agent.number_games)
            print("Score:", score)
            print("Record:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_games
            plot_avg_scores.append(mean_score)
            plot(plot_scores, plot_avg_scores)


if __name__ == '__main__':
    train()
