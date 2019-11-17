from keras.optimizers import Adam
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout
import sys, random
import numpy as np
import pandas as pd
import curses, threading, time
from operator import add
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from simulator import Linefield


class Agent(object):
    def __init__(self, display=False, model_path=None, init_epsilon=0.8, decay_epsilon=0.8):
        if display:
            curses.initscr()
            self.win = curses.newwin(22, 122, 0, 0)  # game area 120*20
            self.win.keypad(True)
            curses.noecho()
            curses.cbreak()
            curses.curs_set(False)
            self.win.border(0)
            self.win.nodelay(True)
        self.game = Linefield()
        self.init_epsilon = init_epsilon
        self.decay_epsilon = decay_epsilon
        self.crash_reward = -100
        self.step_reward = 5
        self.gamma = 0.9
        self.learning_rate = 0.001
        self.field_height, self.field_width = self.game.get_field_size()
        self.field_size = self.field_height * self.field_width
        self.memory = deque(maxlen=2000)
        if model_path is None:
            self.model = self.init_model()
        else:
            self.model = self.load_model(model_path)

    def init_model(self):
        model = Sequential()
        # model.add(Dense(output_dim=120, activation='relu', input_dim=self.field_size))
        model.add(Dense(output_dim=120, activation='relu', input_dim=self.field_height * 3))
        model.add(Dropout(0.15))
        model.add(Dense(activation='relu', units=120))
        model.add(Dropout(0.15))
        model.add(Dense(activation='relu', units=120))
        model.add(Dropout(0.15))
        model.add(Dense(activation='softmax', units=3))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)
        print("Initialized model")

        return model

    def save_model(self, model, save_path='model.json'):
        model_json = model.to_json()
        with open(save_path, "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")
        print("Saved model to {}".format(save_path))

    def load_model(self, model_path):
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        print("Loaded model from {}".format(model_path))

        return loaded_model

    def calculate_step_reward(self, state, is_crush):
        if is_crush:
            return self.crash_reward
        else:
            return self.step_reward + self.gamma * np.amax(self.model.predict(state))

    def train_each_step(self, state, action, reward):
        target = self.model.predict(state)
        target[0][action] = reward
        # print("train each step: target: {}; action: {}".format(target, action))
        self.model.fit(state, target, epochs=1, verbose=0)

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def train_each_round(self):
        if len(self.memory) > 32:
            batch = random.sample(self.memory, 32)
        else:
            batch = self.memory
        for state, action, reward in batch:
            self.train_each_step(state, action, reward)

    def train_model(self, epochs=1000):
        print("Start training...")
        epoch_id = 0
        scores = []
        max_score, max_idx = 0, 0
        while epoch_id < epochs:
            # new iter/game to train the model
            self.game.__init__()
            actions, rewards = [], []
            eps = self.init_epsilon
            game_idx = 0
            while self.game.keep_gaming_flag:
                # update ship by the model
                cur_state = self.get_state()
                if game_idx > 18:
                    eps *= self.decay_epsilon
                # if random.random() < max(self.epsilon, 1 - epoch_id / epochs):
                if random.random() < eps:
                    action = random.randint(0, 2)
                else:
                    pred = self.model.predict(cur_state)
                    # print('pred:', pred)
                    action = np.argmax(pred[0])
                if action == 1:
                    self.game.move_left()
                elif action == 2:
                    self.game.move_right()
                else:
                    # keep current position
                    pass
                new_state = self.get_state()

                # update field
                self.game.update_field()

                reward = self.calculate_step_reward(new_state, self.game.is_crash())
                # if action != 0:
                #     reward += 2
                self.train_each_step(cur_state, action, reward)
                self.remember(cur_state, action, reward)
                actions.append(action)
                rewards.append(reward)

            self.train_each_round()
            epoch_id += 1
            scores.append(self.game.score)
            if self.game.score > max_score:
                max_score = self.game.score
                max_idx = epoch_id
                self.save_model(self.model)
            print("epoch {} reaches score {}".format(epoch_id, self.game.score))
            print("actions: {}".format(actions))
            print("rewards: {}".format(rewards))
        print("epoch {} reaches score {}".format(max_idx, max_score))
        return scores

    def test_model(self):
        # print_thread = threading.Thread(target=self.test_model)
        # print_thread.start()

        while self.game.keep_gaming_flag:
            # update ship
            cur_state = self.get_state()
            pred = self.model.predict(cur_state)
            action = np.argmax(pred[0])
            print("action:", action)
            if action == 1:
                self.game.move_left()
            elif action == 2:
                self.game.move_right()
            else:
                # keep current position
                pass

            # update field
            self.game.update_field()
            self.game.change_speed()

            self.print_state(cur_state)

        # print_thread.join()
        return self.game.score

    def get_state(self):
        # 0: empty, 1: field, 2: ship
        # state = np.zeros((self.field_height, self.field_width), dtype=int)
        # for i in range(self.field_height):
        #     for j in range(self.field_width):
        #         if self.game.field[i][j] == '_':
        #             state[i][j] = 1
        #
        # state[self.game.ship[0]][self.game.ship[1]] = 2
        state = np.zeros((self.field_height, 3), dtype=float)
        i, j = 0, 0
        while i < self.field_height:
            j = 0
            cube = 0
            while j < self.game.ship[1]:
                if self.game.field[i][j] == '_':
                    cube += 1
                j += 1
            state[i][0] = cube / min(1, self.game.ship[1])
            state[i][1] = 1 if self.game.field[i][j] == '_' else 0
            j += 1
            cube = 0
            while j < self.field_width:
                if self.game.field[i][j] == '_':
                    cube += 1
                j += 1
            state[i][2] = cube / min(1, self.game.ship[1])
            i += 1

        return state.reshape((1, -1))

    def print_state(self, state):
        output = "=" * (self.field_width + 2) + '\n'
        state = state.reshape((self.field_height, self.field_width))
        for line in state:
            output += '|'
            for item in line:
                if item == 0:
                    output += ' '
                elif item == 1:
                    output += '_'
                else:
                    output += '^'
            output += '|\n'
        output += "=" * (self.field_width + 2) + '\n'
        print(output)

        # stdscr.addstr(output)
        # stdscr.refresh()

    def print_test_model(self):
        # model_thread = threading.Thread(target=self.test_model)
        # model_thread.start()

        while self.game.keep_gaming_flag:
            self.win.border(0)
            self.win.addstr(0, 2, 'Score : ' + str(self.game.score) + ' ')  # Printing 'Score'
            self.win.addstr(0, 40, ' LineField in Python ')
            self.win.timeout(30)  # refresh rate

            cur_field, ship = self.game.get_game_status()

            # print field, extra 1 is offset for border
            for i in range(0, self.field_height):
                for j in range(0, self.field_width):
                    cur_val = cur_field[i][j]
                    self.win.addch(i + 1, j * 2 + 1, cur_val)
                    self.win.addch(i + 1, j * 2 + 2, cur_val)
            # print ship, extra 1 is offset for border
            self.win.addch(ship[0] + 1, ship[1] * 2 + 1, '/')
            self.win.addch(ship[0] + 1, ship[1] * 2 + 2, '\\')

        # model_thread.join()


def plot_seaborn(x, y):
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([x])[0], np.array([y])[0], color="b", x_jitter=.1, line_kws={'color':'green'})
    ax.set(xlabel='games', ylabel='score')
    plt.show()


if __name__ == '__main__':

    max_epochs = 1000
    agent = Agent()
    scores = agent.train_model(epochs=max_epochs)
    print(scores)
    plot_seaborn(range(max_epochs), scores)

    # agent = Agent(model_path='model.json')
    # score = agent.test_model()
    # print(score)

    # stdscr = curses.initscr()
    # curses.noecho()
    # curses.cbreak()
    # agent = Agent(display=False, model_path='model.json')
    # score = agent.test_model()
    # print(score)


