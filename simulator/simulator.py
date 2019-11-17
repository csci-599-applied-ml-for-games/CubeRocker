from random import randint
import curses, threading, time
from multiprocessing import Process
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN


class Linefield(object):

    def time_step(self, score):
        speed_level = score if score < 19000 else 19000
        return 1000 - speed_level // 20

    def __init__(self):
        self.score = 0
        self._field_height = 20
        self._field_width = 60
        self.field = [[' '] * self._field_width for i in range(self._field_height)]
        self.ship = [19, 29] # row, col
        self.keep_gaming_flag = True

    def generate_each_line(self):
        cur_line = []
        for i in range(0, self._field_width):
            random_val= randint(1, 100)
            if random_val == 20 or random_val == 40 or random_val == 60 or random_val == 80 or random_val == 100:
                cur_line.append('_')
            else:
                cur_line.append(' ')
        return cur_line

    def update_field(self):
        new_line = self.generate_each_line()
        del self.field[self._field_height - 1]
        self.field.insert(0, new_line)
        if self.is_crash():
            self.stop_game()
        else:
            self.score += 1

    def start_game(self):
        while self.keep_gaming_flag:
            self.update_field()
            if self.is_crash():
                print("Game Over, Score: " + str(self.score) + ", `esc` to Exit ")
                return self.ship, self.score

    def change_speed(self):
        time.sleep(0.001)
        # curses.napms(100)
        # curses.napms(self.time_step(self.score))

    def move_left(self):
        self.ship[1] = self.ship[1] - 1 if self.ship[1] > 0 else 0

    def move_right(self):
        self.ship[1] = self.ship[1] + 1 if self.ship[1] < self._field_width - 1 else self._field_width - 1

    def get_game_status(self):
        return self.field, self.ship

    def stop_game(self):
        self.keep_gaming_flag = False

    def get_field_size(self):
        return self._field_height, self._field_width

    def is_crash(self):
        return self.field[self.ship[0]][self.ship[1]] == '_'


class Simulator(object):
    def __init__(self, display=False):
        if display:
            curses.initscr()
            self.win = curses.newwin(22, 122, 0, 0) # game area 120*20
            self.win.keypad(True)
            curses.noecho()
            curses.cbreak()
            curses.curs_set(False)
            self.win.border(0)
            self.win.nodelay(True)

            self.key = KEY_UP
        self.game = Linefield()

    def start_sim(self):

        game_thread = threading.Thread(target=self.game.start_game)
        game_thread.start()

        # p = Process(target=self.game.start_game)
        # p.start()

        while self.key != 27:
            grid, ship = self.game.get_game_status()
            # print(grid[0])
            self.win.border(0)
            self.win.addstr(0, 2, 'Score : ' + str(self.game.score) + ' ')  # Printing 'Score'
            self.win.addstr(0, 40, ' LineField in Python ')
            self.win.timeout(30)  # refresh rate

            event = self.win.getch()
            self.key = self.key if event == -1 else event

            if self.key not in [KEY_LEFT, KEY_RIGHT, 27]:  # If an invalid key is pressed
                self.key = KEY_UP

            if self.key == KEY_LEFT:
                self.game.move_left()
                self.key = KEY_UP
            elif self.key == KEY_RIGHT:
                self.game.move_right()
                self.key = KEY_UP

            cur_field, ship = self.game.get_game_status()

            # print field, extra 1 is offset for border
            for i in range(0, self.game._field_height):
                for j in range(0, self.game._field_width):
                    cur_val = cur_field[i][j]
                    self.win.addch(i + 1, j * 2 + 1, cur_val)
                    self.win.addch(i + 1, j * 2 + 2, cur_val)
            # print ship, extra 1 is offset for border
            self.win.addch(ship[0] + 1, ship[1] * 2 + 1, '/')
            self.win.addch(ship[0] + 1, ship[1] * 2 + 2, '\\')

        game_thread.join()


# main for test
if __name__ == '__main__':
    the_sim = Simulator(display=True)
    the_sim.start_sim()
    curses.endwin()

