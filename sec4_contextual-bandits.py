import tensorflow as tf
import numpy as np


class ContextualBandit:

    def __init__(self):
        self.active_bandit = 0  # state
        self.bandits = np.array([
            [0.2, 0.0, 0.1, -4.0],  # 4th arm best
            [0.1, -5.0, 1.0, 0.25],  # 2nd arm best
            [-3.5, 2.0, 3.2, 6.4]  # 1st arm best
        ])
        self.num_bandits, self.num_actions = self.bandits.shape

    def get_bandit(self):
        self.active_bandit = np.random.randint(0, self.num_bandits)
        return self.active_bandit

    def pull(self, arm):
        bandit = self.bandits[self.active_bandit, arm]
        return 1 if np.random.randn(1) > bandit else -1


class Agent:

    def __init__(self, learning_rate=1e-3, contexts=3, actions=4):
        self.num_actions = actions
        
        self.reward_in = tf.placeholder(tf.float32, [1], name='reward_in')
        self.context_in = tf.placeholder(tf.int32, [1], name='context_in')
        self.action_in = tf.placeholder(tf.int32, [1], name='action_in')

        # sess.run(best_action) to calculate the best action
        context_one_hot = tf.one_hot(self.context_in, contexts)
        W = tf.get_variable('W', [contexts, actions])
        self.output = tf.nn.sigmoid(tf.matmul(context_one_hot, W))
        self.best_action = tf.argmax(self.output, axis=1)

        # sess.run(optimizer) to update the best action
        a_ = tf.reduce_sum(self.output * tf.one_hot(self.action_in, actions))
        self.loss = -(tf.log(a_) * self.reward_in)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.loss)

    def predict(self, sess, context):
        return sess.run(self.best_action, {self.context_in: [context]})[0]

    def random_or_predict(self, sess, epsilon, context):
        if np.random.rand(1) < epsilon:
            return np.random.randint(self.num_actions)
        else:
            return self.predict(sess, context)

    def train(self, sess, context, action, reward):
        sess.run(self.optimizer, {
            self.action_in: [action],
            self.reward_in: [reward],
            self.context_in: [context]
        })


env = ContextualBandit()
agent = Agent()
num_episodes = 30000
epsilon = 0.1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for ep in range(num_episodes):
        context = env.get_bandit()
        action = agent.random_or_predict(sess, epsilon, context)
        reward = env.pull(action)
        # feed state, action, reward back to the policy network
        agent.train(sess, context, action, reward)
        if ep % 500 == 0:
            loss = sess.run(agent.loss, {
                agent.action_in: [action],
                agent.reward_in: [reward],
                agent.context_in: [context]
            })
            print('Step {}, Loss={}'.format(ep, loss))
    
    # results time
    print(np.argmin(env.bandits, axis=1))
    print('Best arm for Bandit 1:')
    print(agent.predict(sess, 0))

    print('Best arm for Bandit 2:')
    print(agent.predict(sess, 1))

    print('Best arm for Bandit 3:')
    print(agent.predict(sess, 2))

