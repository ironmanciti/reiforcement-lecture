"""
강화학습 maze 예제
이 script 는 q-learning algorithm 을 이용하여 table 을 update 하는 module 들을 control 한다.
RL algorthm (Q-Learning) 은 RL_agent_df.py (pandas dataframe 으로 Q-table 구성) 혹은 RL_agent_dict.py
(Python default dict 로 q-table 구성) 이다.
environment 는 maze-env.py 이다.
Red 사각형 - agent
Black 사각형 - hell (reward -1)
Yellow circle - paradise (reward +1)
다른 모든 states - reward 0
"""

from maze_env import Maze 
from RL_agent_dict import QLearningTable
#from RL_agent_df import QLearningTable
import matplotlib
import matplotlib.pyplot as plt

episode_cnt = 50  # number of episodes to run the experiment
rewards = []      # 각 episode 의 reward 저장
movements = []    # 각 episode 에서 수행된 이동 횟수

"""
이 함수는 agent 가 선택한 action 에 따라 maze 환경에서 agent 의 위치를 update 한다.
"""
def run_experiment():

    for episode in range(episode_cnt):
        print("Episode {}/{}".format(episode, episode_cnt))
        obeservation = env.reset()
        moves = 0

        while True:
            env.render()
            # Q-learning 은 observation 에 의해 action 을 선택한다
            # key 값으로 사용하기 위해 observation 을 string 으로 변환
            action = q_learning_agent.choose_action(str(obeservation))
            obeservation_, reward, done = env.get_state_reward(action)
            moves += 1

            # 위의 transition 에 따라 주어진 tuple(s, a, r, s') 을 가지고 q-table update 
            q_learning_agent.learn(str(obeservation), action, reward, str(obeservation_))
            obeservation = obeservation_

            if done:
                movements.append(moves)
                rewards.append(reward)
                print("Reward: {}, Moves: {}".format(reward, moves))
                break

    print("The game is over !")

    plot_reward_movements()

def plot_reward_movements():
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(episode_cnt), movements)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")

    plt.subplot(2, 1, 2)
    plt.plot(range(episode_cnt), rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")

    plt.savefig("reward_movement_qlearning.png")

    plt.show()

if __name__ == "__main__":
    env = Maze()

    #q-learning agent 생성
    q_learning_agent = QLearningTable(actions=list(range(env.n_actions)))

    # 10 millisecond 지연 후 run_experiment 수행
    env.window.after(10, run_experiment)
    # window 가 close 될 때까지 application 을 무한 loop
    env.window.mainloop()
