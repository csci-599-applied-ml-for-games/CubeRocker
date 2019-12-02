from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
import random
import numpy as np
import pandas as pd
from operator import add


class DQNAgent(object):

    def __init__(self, output_dim=3):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.001
        self.epsilon = 0
        self.actual = []
        self.memory = []

        self.input_dimension = 22 * 22
        self.hidden_dimension = 120 
        self.output_dimension = output_dim # number of moves
        self.model = self.network()


    def set_reward(self, player, crash, action):
        self.reward = 0
        if crash:
            self.reward = -200
            return self.reward
        elif action == 0: # no move
            self.reward = 1
        else:
            self.reward = 5
        return self.reward

    def get_state(self, game, player, field):
        # return a vector of int to represent the position
        # 0: empty, 1: field, 2: ship

        state = np.zeros((field.height, field.width), dtype=int)
        for i in range(field.height):
            for j in range(field.width):
                if field.grid[i][j] == 1:
                    state[i][j] = 1

        state[player.x][player.y] = 2

        return state.reshape((1, 22 * 22))
        # return np.asarray(state)

    def network(self, weights=None):
        # linear model
        print(self.output_dimension)

        model = Sequential()
        model.add(Dense(output_dim=self.hidden_dimension, activation='relu', input_dim=self.input_dimension))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=self.hidden_dimension, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=self.hidden_dimension, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=self.output_dimension, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state))# np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(state)
            target_f[0][np.argmax(action)] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][np.argmax(action)] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)