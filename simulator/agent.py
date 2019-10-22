from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
import curses, threading, time
from operator import add
from simulator import Linefield


class Agent(object):
    def __init__(self, display=False, model_path=None, eps=0.01):
        if display:
            pass
        self.game = Linefield()
        self.epsilon = eps
        self.crash_reward = -25
        self.step_reward = 25
        self.gamma = 0.9
        self.learning_rate = 0.01
        self.field_height, self.field_width = self.game.get_field_size()
        self.field_size = self.field_height * self.field_width
        if model_path is None:
            self.model = self.init_model()
        else:
            self.model = self.load_model(model_path)

    def init_model(self):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=self.field_size))
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

    def train_model(self, epochs=100):
        print("Start training...")
        epoch_id = 0
        scores = []
        while epoch_id < epochs:
            # new iter/game to train the model
            self.game.__init__()
            while self.game.keep_gaming_flag:
                # update ship
                cur_state = self.get_state()
                if random.random() < max(self.epsilon, 1 - epoch_id / epochs):
                    action = random.randint(0, 2)
                else:
                    pred = self.model.predict(cur_state)
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
                if self.game.is_crash():
                    reward = self.crash_reward
                else:
                    self.game.score += 25
                    reward = self.step_reward + self.gamma * np.amax(self.model.predict(new_state)[0])
                target = self.model.predict(cur_state)
                target[0][np.argmax(action)] = reward
                self.model.fit(cur_state, target, epochs=1, verbose=0)
            epoch_id += 1
            scores.append(self.game.score)
            print("epoch {} reaches score {}".format(epoch_id, self.game.score))
        return scores

    def test_model(self):
        while self.game.keep_gaming_flag:
            # update ship
            cur_state = self.get_state()
            if random.random() < max(self.epsilon, 1 - epoch_id / epochs):
                action = randint(0, 2)
            else:
                pred = self.model.predict(cur_state)
                action = np.argmax(pred[0])
            if action == 1:
                self.game.move_left()
            elif action == 2:
                self.game.move_right()
            else:
                # keep current position
                pass

            # update field
            self.game.update_field()
            if self.is_crash():
                print("Game Over, Score: " + str(self.game.score) + ", `esc` to Exit ")
                return self.game.ship, self.game.score
            self.game.score += 25
            self.game.change_speed()

    def get_state(self):
        # 0: empty, 1: field, 2: ship
        state = np.zeros((self.field_height, self.field_width), dtype=int)
        for i in range(self.field_height):
            for j in range(self.field_width):
                if self.game.field[i][j] == '_':
                    state[i][j] = 1

        state[self.game.ship[0]][self.game.ship[1]] = 2

        return state.reshape((1, self.field_size))


if __name__ == '__main__':
    agent = Agent()
    scores = agent.train_model(epochs=5)
    print(scores)

