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
        self.crash1 = False
        self.crash2 = False
        self.player1 = Player(self, x=4)
        self.player2 = Player(self, x=14)
        self.field = Field(game_width, game_height)
        

class Player(object):

    def __init__(self, game, x=9, y=21):
        # 22 block in field, each block 20 px
        # x = int(0.45 * game.game_width / 20)
        # y = int((game.game_height - 20) / 20)
        self.x = x
        self.y = y
        self.image = pygame.image.load('img/food2.png')
        self.display = True
        self.score = 0

    def update_position(self, x, y):
        self.x = x
        self.y = y

    def do_move(self, move, field, game, players):
        if move == 1: # left
            if self.x > 0:
                self.x -= 1
                move = 0
        elif move == 2:  # right
            if self.x < int(game.game_width / 20) - 2:
                self.x += 1
                move = 0
        elif move == 3: # up
            if self.y < int(game.game_height / 20) - 2:
                self.y += 1
                move = 0
        elif move == 4: # down
            if self.y > 0:
                self.y -= 1
                move = 0

        field.field_update(game)
        
        if_crash = False
        if field.grid[self.y][self.x] == 1:
            if_crash = True
        else:
            for p in players:
                if p.x == self.x and p.y == self.y and p.display:
                    if_crash = True
                    break
        
        if if_crash and self.display:
            game.gameDisplay.blit(field.explode, (self.x * 20, (self.y) * 20))
            update_screen()
            self.display = False
        elif self.display:
            self.score += 25

        

        pygame.time.wait(300)

        return if_crash

    def display_player(self, game):

        if self.display:
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

    def field_update(self, game):
        new_line = self.generate_new_line(game)
        self.grid.insert(0, new_line)
        self.grid.pop()
        # if game.crash1 == False and self.grid[player1.y][player1.x] == 1:# grid is height first
        #     game.crash1 = True
        #     game.gameDisplay.blit(self.explode, (player1.x * 20, (player1.y) * 20))
        # if game.crash2 == False and self.grid[player2.y][player2.x] == 1:# grid is height first
        #     game.crash2 = True
        #     game.gameDisplay.blit(self.explode, (player2.x * 20, (player2.y) * 20))
        game.crash = game.crash1 and game.crash2
        update_screen()

    def display_field(self, game):
        for i in range(0, self.height):
            for j in range(0, self.width):
                if self.grid[i][j] == 1:
                    game.gameDisplay.blit(self.image, (j * 20, i * 20)) # grid is height first
        update_screen()

    def generate_new_line(self, game):
        cur_line = []
        for i in range(0, self.width):
            random_val= randint(0, 100)
            # set the maximum number of cubes in the new line according to the current score
            s = max(game.player1.score, game.player2.score)
            if s < 1000:
                num_cubes = 1
            elif s < 1500:
                num_cubes = 2
            elif s < 2000:
                num_cubes = 3
            elif s < 2500:
                num_cubes = 4
            else:
                num_cubes = 5
            if random_val < num_cubes:
                cur_line.append(1)
            else:
                cur_line.append(0)

        return cur_line


def get_record(score1, score2, record):
        return max([score1, score2, record])


def display_ui(game, score1, score2, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score1 = myfont.render('SCORE 1: ', True, (0, 0, 0))
    text_score1_number = myfont.render(str(score1), True, (0, 0, 0))
    text_score2 = myfont.render('SCORE 2: ', True, (0, 0, 0))
    text_score2_number = myfont.render(str(score2), True, (0, 0, 0))
    game.gameDisplay.blit(text_score1, (45, 440))
    game.gameDisplay.blit(text_score1_number, (120, 440))
    game.gameDisplay.blit(text_score2, (190, 440))
    game.gameDisplay.blit(text_score2_number, (350, 440))
    game.gameDisplay.blit(game.bg, (10, 10))


def display(player1, player2, field, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, player1.score, player2.score, record)
    field.display_field(game)
    player1.display_player(game)
    player2.display_player(game)


def update_screen():
    pygame.display.update()


def initialize_game(player1, player2, game, field, agent):
    # state_init1 = agent.get_state(game, player1, field)  
    action = 0
    if player1.do_move(action, field, game, [player2]):
        game.crash1 = True
    if player2.do_move(action, field, game, [player1]):
        game.crash2 = True
    game.crash = game.crash1 and game.crash2
    # state_init2 = agent.get_state(game, player, field)
    # reward1 = agent.set_reward(player, game.crash, action)
    # agent.remember(state_init1, action, reward1, state_init2, game.crash)
    # agent.replay_new(agent.memory)


def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1, line_kws={'color':'green'})
    ax.set(xlabel='games', ylabel='score')
    plt.show()


def train(epoch=10):
    pygame.init()
    agent = DQNAgent(output_dim=5)
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    while counter_games < epoch:
        # Initialize classes
        game = Game(440, 440)
        player1 = game.player1
        player2 = game.player2
        field0 = game.field


        # Perform first move
        initialize_game(player1, player2, game, field0, agent)
        if display_option:
            display(player1, player2, field0, game, record)

        game_epoch = 0
        while not game.crash:
            #agent.epsilon is set to give randomness to actions
            agent.epsilon = 50 - game_epoch

            train_each_epoch(agent, game, field0, player1, [player2], game_epoch)
            train_each_epoch(agent, game, field0, player2, [player1], game_epoch)

            record = get_record(game.player1.score, game.player2.score, record)
            if display_option:
                display(player1, player2, field0, game, record)
                pygame.time.wait(speed)
            
            game_epoch += 1
            game.crash = not (game.player1.display or game.player2.display)

        agent.replay_new(agent.memory)
        counter_games += 1
        print('Game', counter_games, '      Score:', game.player1.score, game.player2.score)
        score_plot.append(game.player1.score)
        counter_plot.append(counter_games)
    agent.model.save_weights('weights_multi.hdf5')
    plot_seaborn(counter_plot, score_plot)


def train_each_epoch(agent, game, field0, player, players, game_epoch):
    #get old state
    state_old = agent.get_state(game, player, field0, players=players)

    #perform random actions based on agent.epsilon, or choose the action
    if randint(0, 100) < agent.epsilon:
        final_move = randint(0, 4)
        print("random with prob {}".format(agent.epsilon))
    else:
        # predict action based on the old state
        prediction = agent.model.predict(state_old)
        final_move = np.argmax(prediction[0])
        print("prediction : {}".format(prediction))
    print("move: {} to position ({}, {})".format(final_move, player.x, player.y))

    #perform new move and get new state
    player.do_move(final_move, field0, game, players)
    state_new = agent.get_state(game, player, field0, players=players)

    if game_epoch >= 19:
        #set treward for the new state
        reward = agent.set_reward(player, game.crash, final_move)

        #train short memory base on the new action and state
        agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)

        # store the new data into a long term memory
        if game.crash:
            agent.remember(state_old, final_move, reward, state_new, game.crash)
            print("remember this move with reward {}".format(reward))
        elif final_move == 0 and randint(1, 20) < 1:
            agent.remember(state_old, final_move, reward, state_new, game.crash)
            print("remember this move with reward {}".format(reward))
        elif final_move != 0 and randint(1, 20) < 5:
            agent.remember(state_old, final_move, reward, state_new, game.crash)
            print("remember this move with reward {}".format(reward))


def test():
    pygame.init()
    agent = DQNAgent(output_dim=5)
    agent.model.load_weights('weights_multi.hdf5')
    # while counter_games < 150:
    # Initialize classes
    game = Game(440, 440)
    player1 = game.player1
    player2 = game.player2
    field0 = game.field

    # Perform first move
    record = 0
    initialize_game(player1, player2, game, field0, agent)
    if display_option:
        display(player1, player2, field0, game, record)

    while not game.crash:
        # get old state
        state_old = agent.get_state(game, player1, field0)

        # predict action based on the old state
        prediction = agent.model.predict(state_old)
        final_move = np.argmax(prediction[0])
        print("move {} with prediction : {}".format(final_move, prediction))

        # perform new move and get new state
        player1.do_move(final_move, field0, game, [player2])

        record = get_record(game.player1.score, game.player2.score, record)
        if display_option:
            display(player1, player2, field0, game, record)
            pygame.time.wait(speed)


if __name__ == "__main__":
    # Set options to activate or deactivate the game view, and its speed
    display_option = True
    speed = 0
    pygame.font.init()
    if sys.argv[1] == "train":
        train(epoch=int(sys.argv[2]))
    else:
        test()
