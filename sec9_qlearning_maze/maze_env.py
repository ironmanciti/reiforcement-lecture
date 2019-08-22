"""
Red 사각형 - agent
Black 사각형 - hell (reward -1)
Yellow circle - paradise (reward +1)
다른 모든 states - reward 0
"""

import numpy as np
import time
import sys
import tkinter as tk

UNIT = 40
MAZE_H = 6
MAZE_W = 6

class Maze:
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r']   # possible actions
        self.n_actions = len(self.action_space)
        self.build_maze()

    def build_maze(self):
        self.window = tk.Tk()  # main window create
        self.window.title("MAZE with Q-Learning")
        self.window.geometry('{}x{}'.format(MAZE_W*UNIT, MAZE_H*UNIT)) # window size
        # canvas create
        self.canvas = tk.Canvas(self.window, bg='white', width=MAZE_W*UNIT, height=MAZE_H*UNIT) 
        # horizontal line
        for c in range(0, MAZE_W*UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_W*UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        # vertical line
        for r in range(0, MAZE_H*UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H*UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        
        self.origin = np.array([20, 20])

        # create 2 hell points
        hell1_center = self.origin + np.array([UNIT*2, UNIT])   # 2 UNIT RIGHT, 1 UNIT DOWN
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15, fill='black'
        )
        hell2_center = self.origin + np.array([UNIT, UNIT*2])   # 1 UNIT RIGHT, 2 UNIT DOWN
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15, fill='black'
        )
        # create goal center
        goal_center = self.origin + UNIT*2   # 2 UNIT RIGHT, 2 UNIT DOWN
        self.goal = self.canvas.create_oval(
            goal_center[0] - 15, goal_center[1] - 15,
            goal_center[0] + 15, goal_center[1] + 15, fill='yellow'
        )

        self.draw_rect()
        # pack all
        self.canvas.pack()
    
    def draw_rect(self):
        self.rect = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15, fill='red'
        )

    def render(self):
        time.sleep(0.1)  # 0.1 초 정지
        self.window.update()

    def reset(self):   # maze reset
        self.window.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        self.draw_rect()

        return self.canvas.coords(self.rect)

    def get_state_reward(self, action):  # action - 0, 1, 2, 3 (up, down, right, left)
        s = self.canvas.coords(self.rect)   # current state
        base_action = np.array([0, 0])      # [x, y] pixel 만큼 이동하기 위한 array
        if action == 0:   # up
            if s[1] > UNIT:              # y 좌표가 UNIT 보다 크면 y 좌표 - UNIT
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:   # y 좌표가 MAZE_H 보다 UNIT 이상 작으면 Y 좌표+UNIT
                base_action[1] += UNIT 
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:   # X 좌표가 MAZE_H 보다 UNIT 이상 작으면 X 좌표+UNIT
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:                  # x 좌표가 UNIT 보다 크면 X 좌표-UNIT
                base_action[0] -= UNIT 
        else:
            print("invalid action")

        self.canvas.move(self.rect, base_action[0], base_action[1])

        s_ = self.canvas.coords(self.rect)   # next state - next coordinate (x, y) 좌표

        if s_ == self.canvas.coords(self.goal):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False
        return s_, reward, done 


if __name__ == '__main__':
    maze = Maze()
    maze.window.mainloop()
