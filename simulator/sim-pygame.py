import pygame, sys
from random import randint
from DQN import DQNAgent
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Game:

    def __init__(self, game_width, game_height):
        pygame.display.set_caption('CubeField')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height+60))
        self.bg = pygame.image.load("img/background2.png")
        self.crash = False
        self.player = Player(self)
        self.field = Field(game_width, game_height)
        self.score = 0
        

class Player(object):

    def __init__(self, game):
        # 22 block in field, each block 20 px
        x = int(0.45 * game.game_width / 20)
        y = int((game.game_height - 20) / 20)
        self.x = x
        self.y = y
        # self.position = []
        # self.position.append([self.x, self.y])
        # self.food = 1
        # self.eaten = False
        self.image = pygame.image.load('img/food2.png')

    def update_position(self, x, y):
        self.x = x
        self.y = y

    def do_move(self, move, field, game):
        # if self.eaten:

        #     self.position.append([self.x, self.y])
        #     self.eaten = False
        #     self.food = self.food + 1
        if move == 1: # left
            if self.x > 0:
                self.x -= 1
                move = 0
        elif move == 2:  # right
            if self.x < int(game.game_width / 20) - 2:
                self.x += 1
                move = 0
        field.field_update(game, self)
        pygame.time.wait(300)

    def display_player(self, game):

        if game.crash == False:
            game.gameDisplay.blit(self.image, (self.x * 20, self.y * 20))
            update_screen()
        else:
            pygame.time.wait(30)


class Field(object):

    def __init__(self, game_width, game_height):
        self.width, self.height = int(game_width / 20), int(game_height / 20)
        self.grid = [[0] * self.width for i in range(self.height)] # grid is height first
        self.image = pygame.image.load('img/snakeBody.png')
        self.explode = pygame.image.load("img/explode.png")

    def field_update(self, game, player):
        new_line = self.generate_new_line(player, game)
        self.grid.insert(0, new_line)
        self.grid.pop()
        if self.grid[player.y][player.x] == 1:# grid is height first
            game.crash = True
            game.gameDisplay.blit(self.explode, (player.x * 20, (player.y) * 20))
            update_screen()

    def display_field(self, game):
        for i in range(0, self.height):
            for j in range(0, self.width):
                if self.grid[i][j] == 1:
                    game.gameDisplay.blit(self.image, (j * 20, i * 20)) # grid is height first
        update_screen()

    def generate_new_line(self, player, game):
        cur_line = []
        for i in range(0, self.width):
            random_val= randint(0, 100)
            # set the maximum number of cubes in the new line according to the current score
            if game.score < 1000:
                num_cubes = 1
            elif game.score < 1500:
                num_cubes = 2
            elif game.score < 2000:
                num_cubes = 3
            elif game.score < 2500:
                num_cubes = 4
            else:
                num_cubes = 5
            if random_val in range(num_cubes):
                cur_line.append(1)
            else:
                cur_line.append(0)
        game.score = game.score + 25
        return cur_line


def get_record(score, record):
        if score >= record:
            return score
        else:
            return record


def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (10, 10))


def display(player, field, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    field.display_field(game)
    player.display_player(game)


def update_screen():
    pygame.display.update()


def initialize_game(player, game, field, agent):
    state_init1 = agent.get_state(game, player, field)  
    action = 0
    player.do_move(action, field, game)
    state_init2 = agent.get_state(game, player, field)
    reward1 = agent.set_reward(player, game.crash, action)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory)


def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1, line_kws={'color':'green'})
    ax.set(xlabel='games', ylabel='score')
    plt.show()


def train(epoch=10):
    pygame.init()
    agent = DQNAgent(output_dim=3)
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    while counter_games < epoch:
        # Initialize classes
        game = Game(440, 440)
        player1 = game.player
        field0 = game.field


        # Perform first move
        initialize_game(player1, game, field0, agent)
        if display_option:
            display(player1, field0, game, record)

        game_epoch = 0
        while not game.crash:
            #agent.epsilon is set to give randomness to actions
            agent.epsilon = 50 - game_epoch

            #get old state
            state_old = agent.get_state(game, player1, field0)

            #perform random actions based on agent.epsilon, or choose the action
            if randint(0, 100) < agent.epsilon:
                final_move = randint(0, 4)
                # print("random with prob {}".format(agent.epsilon))
            else:
                # predict action based on the old state
                prediction = agent.model.predict(state_old)
                final_move = np.argmax(prediction[0])
                # print("prediction : {}".format(prediction))
            # print("move: {} to position ({}, {})".format(final_move, player1.x, player1.y))

            #perform new move and get new state
            player1.do_move(final_move, field0, game)
            state_new = agent.get_state(game, player1, field0)

            if game_epoch >= 19:
                #set treward for the new state
                reward = agent.set_reward(player1, game.crash, final_move)

                #train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)

                # store the new data into a long term memory
                if game.crash:
                    agent.remember(state_old, final_move, reward, state_new, game.crash)
                    # print("remember this move with reward {}".format(reward))
                elif final_move == 0 and randint(1, 20) < 1:
                    agent.remember(state_old, final_move, reward, state_new, game.crash)
                    # print("remember this move with reward {}".format(reward))
                elif final_move != 0 and randint(1, 20) < 5:
                    agent.remember(state_old, final_move, reward, state_new, game.crash)
                    # print("remember this move with reward {}".format(reward))

            record = get_record(game.score, record)
            if display_option:
                display(player1, field0, game, record)
                pygame.time.wait(speed)
            
            game_epoch += 1

        agent.replay_new(agent.memory)
        counter_games += 1
        print('Game', counter_games, '      Score:', game.score)
        score_plot.append(game.score)
        counter_plot.append(counter_games)

        if game.score >= record:
            agent.model.save_weights(modelFile + '/weights.hdf5')
    agent.model.save_weights(modelFile + '/weightsFinal.hdf5')
    plot_seaborn(counter_plot, score_plot)


def test():
    pygame.init()
    agent = DQNAgent(output_dim=3)
    agent.model.load_weights('weights.hdf5')
    # while counter_games < 150:
    # Initialize classes
    game = Game(440, 440)
    player1 = game.player
    field0 = game.field

    # Perform first move
    record = 0
    initialize_game(player1, game, field0, agent)
    if display_option:
        display(player1, field0, game, record)

    while not game.crash:
        # get old state
        state_old = agent.get_state(game, player1, field0)

        # predict action based on the old state
        prediction = agent.model.predict(state_old)
        final_move = np.argmax(prediction[0])
        print("move {} with prediction : {}".format(final_move, prediction))

        # perform new move and get new state
        player1.do_move(final_move, field0, game)

        record = get_record(game.score, record)
        if display_option:
            pygame.time.wait(speed)
            display(player1, field0, game, record)


if __name__ == "__main__":
    # Set options to activate or deactivate the game view, and its speed
    display_option = False
    speed = 0
    pygame.font.init()
    modelFile = sys.argv[3]
    if len(sys.argv) < 4:
        print("Usage: python sim-py-game-<version>.py train/test numberOfEpoch modelFolder")
        exit
    if sys.argv[1] == "train":
        train(epoch=int(sys.argv[2]))
    else:
        test()