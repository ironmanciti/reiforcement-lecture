# Suntton p.92 First-Visit MC predictions, for estimating V ~ v_pi
# card 조합에 따른 value estimation

import gym 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

SCORE = 0  # state 내의 player 의 카드 합계 index
# ex) (6, 1, False) => (sum_hand(player), dealer open card, usable_ace 보유)
# episode 내의 array element 순서, [next_state, action, reward]
STATE = 0
ACTION = 1
REWARD = 2

def generate_episode(policy, env):
    episode = []
    cur_state = env.reset()
    while True:  # openAi gym 에서 termination 받을 때까지 episode 계속
        action = policy(cur_state)
        next_state, reward, done, _ = env.step(action)  # state, reward, done, {}
        # state : ex) (6, 1, False) => (sum_hand(player), dealer open card, usable_ace 보유)
        episode.append([next_state, action, reward])
        if done:
            return episode 
        cur_state = next_state

def first_visit_MC_evaluate(policy, env, num_episodes):
    V = defaultdict(float)
    returns = defaultdict(list)

    for _ in range(num_episodes):
        episode = generate_episode(policy, env)
        appeared_states = set([step[STATE] for step in episode])
        for s in appeared_states:
            first_occurance_of_s = next(idx for idx, step in enumerate(episode)
                                        if step[STATE] == s)
            # first visit 이후의 모든 return 을 평균함
            G = sum([step[REWARD] for step in episode[first_occurance_of_s:]])
            returns[s].append(G)
            V[s] = np.mean(returns[s])
    return V 

def policy(state):
    return 0 if state[SCORE] >= 20 else 1

if __name__ == "__main__":
    env = gym.make("Blackjack-v0")
    V = first_visit_MC_evaluate(policy, env, num_episodes=100000)
    
    X, Y = np.meshgrid(
            np.arange(12, 22), # player 가 가진 card 합계
            np.arange(1, 11)   # dealer 가 open 한 card
            )
    # np.dstack([X, Y]) -> axis=2 추가
    no_usable_ace = np.apply_along_axis(lambda idx: V[(idx[0], idx[1], False)], 2, np.dstack([X, Y]))
    usable_ace = np.apply_along_axis(lambda idx: V[(idx[0], idx[1], True)], 2, np.dstack([X, Y]))
    
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4), subplot_kw={'projection': '3d'})
    ax0.plot_surface(X, Y, no_usable_ace, cmap=plt.cm.YlGnBu_r)
    ax0.set_xlabel('Hole Cards')
    ax0.set_ylabel('Dealer')
    ax0.set_zlabel('MC Estimated Value')
    ax0.set_title('No Useable Ace')
    
    ax1.plot_surface(X, Y, usable_ace, cmap=plt.cm.YlGnBu_r)
    ax1.set_xlabel('Hole Cards')
    ax1.set_ylabel('Dealer')
    ax1.set_zlabel('MC Estimated Value')
    ax1.set_title('Useable Ace')
    
    plt.show()
