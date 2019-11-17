"""
    10-Armed Testbed (Reinforcement Learning: An Introduction, Sutton, Barto, fig 2.2)
    Created by Jet-Tsyn Lee 23/02/18, last update 03/01/18
    Program is to compare the greedy and e-greedy methods in a 10-armed bandit testbed, presented
    in the Reinforcement Learning: An Introduction book, Sutton, Barto, fig 2.2.
"""
import numpy as np
import matplotlib.pyplot as plt
import time

################################################################
# TestBed class containing the states and actions, and the overall rules of the test
class KarmedBandit(object):

    # Constructor
    def __init__(self, nArms, mean, stDev):

        # Number of arms
        self.nArms = nArms

        # Used for the Gaussian random actions
        self.mean = mean        # Mean
        self.stDev = stDev      # Standard Deviation

        self.reset()

    # Reset bandit for next iteration
    def reset(self):
        # 정규분포로 10 개의 arm (action array) 을 reset
        self.actArr = np.random.normal(self.mean, self.stDev, self.nArms)
        # maximum action value 를 갖는 arm
        self.optim = np.argmax(self.actArr)

###############################################################
# Agent Class - Controls the agents action in the environment interacting with the bandit
class Agent(object):

    # Constructor
    def __init__(self, nArms, epsilon=0):
        self.nArms = nArms         # Number of arms to select
        self.epsilon = epsilon     # Epsilon
        self.timeStep = 0                    # Time Step t
        self.lastAction = None               # Store last action

        self.kAction = np.zeros(nArms)       # count of actions taken at time t
        self.rSum = np.zeros(nArms)          # Sums number of rewards
        self.valEstimates = np.zeros(nArms)  # action value estimates sum(rewards)/Amount -> Qt ~= Q*(a)

    # Return string for graph legend
    def __str__(self):
        if self.epsilon == 0:
            return "Greedy"
        else:
            return "Epsilon = " + str(self.epsilon)

    # Selects action based on a epsilon-greedy behaviour
    def action(self):
        ### POLICY ###
        randProb = np.random.random()
        if randProb < self.epsilon:   # Epsilon method
            a = np.random.choice(self.nArms)    # 무작위로 arm 선택
        else:  # Greedy method
            maxAction = np.argmax(self.valEstimates)  # Find max value estimate
            # 같은 Value Estimation 의 action 이 여러개 있을 수 있음을 감안
            action = np.where(self.valEstimates == np.argmax(self.valEstimates))[0]

            # If multiple actions contain the same value, randomly select an action
            if len(action) == 0:
                a = maxAction
            else:
                a = np.random.choice(action)

        # save last action in variable, and return result
        self.lastAction = a
        return a


    # Interpreter - updates the value extimates amounts based on the last action
    def interpreter(self, reward):
        # Add 1 to the number of action taken in step
        At = self.lastAction

        self.kAction[At] += 1       # Add 1 to action selection
        self.rSum[At] += reward     # Add reward to sum array

        # Calculate new action-value, sum(r)/ka
        self.valEstimates[At] = self.rSum[At]/self.kAction[At]

        # Increase time step
        self.timeStep += 1


    # Reset all variables for next iteration
    def reset(self):
        self.timeStep = 0
        self.lastAction = None

        self.kAction[:] = 0
        self.rSum[:] = 0
        self.valEstimates[:] = 0


nArms = 10     # n number of bandits
bandit = KarmedBandit(nArms=nArms, mean=0, stDev=3)

start_time = time.time()    #store time to monitor execution

agents = [Agent(nArms=nArms), Agent(nArms=nArms, epsilon=0.1), Agent(nArms=nArms, epsilon=0.01)]

iterations = 2000           # number of repeated iterations
plays = 1000                # number of plays per iteration

# Array to store the scores, number of plays X number of agents
scoreArr = np.zeros((plays, len(agents)))
# Array to maintain optimal count, Graph 2
optimlArr = np.zeros((plays, len(agents)))

# loop for number of iterations
for iIter in range(iterations):

    if (iIter%100) == 0:   # Print statement after every 100 iterations
        print("Completed Iterations: ", iIter)

    bandit.reset()   #Reset testbed and all agents
    for agent in agents:
        agent.reset()

    # Loop for number of plays
    for jPlays in range(plays):

        for i, kAgent in enumerate(agents):
            actionT =  kAgent.action()

            # Reward - normal dist (mean = Q*(a_t), variance = 1)
            rewardT = np.random.normal(bandit.actArr[actionT], scale=1)

            # Agent checks state
            kAgent.interpreter(reward=rewardT)

            # Add score in arrary, graph 1
            scoreArr[jPlays, i] += rewardT

            # check the optimal action, add optimal to array, graph 2
            if actionT == bandit.optim:
                optimlArr[jPlays, i] += 1


scoreAvg = scoreArr/iterations
optimlAvg = optimlArr/iterations

print("Execution time: %s seconds" % (time.time() - start_time))

#Graph 1 - Averate rewards over all plays
plt.title("10-Armed TestBed - Average Rewards")
plt.plot(scoreAvg)
plt.ylabel('Average Reward')
plt.xlabel('Plays')
plt.legend(agents, loc=4)
plt.show()

#Graph 1 - optimal selections over all plays
plt.title("10-Armed TestBed - % Optimal Action")
plt.plot(optimlAvg  * 100)
plt.ylim(0, 100)
plt.ylabel('% Optimal Action')
plt.xlabel('Plays')
plt.legend(agents, loc=4)
plt.show()






