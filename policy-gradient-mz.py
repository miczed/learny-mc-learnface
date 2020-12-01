import gym
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from gym import logger as gymlogger
from gym.wrappers import Monitor
from gym.wrappers import TimeLimit
gymlogger.set_level(40)  # error only



class PolicyGradientNetwork(tf.keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(PolicyGradientNetwork, self).__init__()
        # is an integer since we're dealing with a discrete action space (policy gradient methods are best for this)
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        # layers
        self.fc1 = tf.keras.layers.Dense(self.fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.fc2_dims, activation='relu')

        # will hold the policy, a probability distribution (sum = 1)
        self.pi = tf.keras.layers.Dense(self.n_actions, activation='softmax')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        pi = self.pi(value)
        return pi


class Agent:
    # alpha = learning rate
    # gamma = discount factor
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=4, fc1_dims=256, fc2_dims=256):
        self.gamma = gamma
        self.alpha = alpha
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy = PolicyGradientNetwork(n_actions=n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.policy.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha))
        self.training_step = 0

    def choose_action(self, observation):
        state = tf.convert_to_tensor(observation, dtype=tf.float32)
        probs = self.policy(state)
        # categorical is used here since we have a set of discrete classes with some probability defined by probs
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()

        # convert tensorflow tensor back to numpy array for our environment
        return action.numpy()[0][0]

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        # convert the actions and rewards to a tensorflow tensor
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_memory)

        # calculate the discounted sum of future rewards for each timestep
        # discounting it ensures that the sum is finite in an infinite setting
        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                # sum of all rewards with discount applied
                G_sum += rewards[k] * discount
                # increasing the discount factor at each timestep
                discount *= self.gamma
                G[t] = G_sum
        # allows us to calculate the gradient
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                # gradient ascent algorithm because we want to maximize total score
                loss += -g * tf.squeeze(log_prob)  # squeeze it to get it back to a scalar quantity
        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))

        # clear memory
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []


def continuize_action(action):
    if action == 0: action = [0, 0, 0.0]  # Nothing
    if action == 1: action = [-1, 0, 0.0]  # Left
    if action == 2: action = [+1, 0, 0.0]  # Right
    if action == 3: action = [0, +1, 0.0]  # Accelerate
    if action == 4: action = [0, 0, 0.8]  # break
    return action


# environment inspired by:
# https://github.com/xtma/pytorch_car_caring/blob/master/train.py
class Env():
    """
    Environment wrapper for CarRacing
    """

    def __init__(self, repeat_actions=8, img_stack=4):
        self.env = gym.make('CarRacing-v0')
        self.reward_threshold = self.env.spec.reward_threshold
        self.repeat_actions = repeat_actions
        self.img_stack = img_stack

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.preprocess(img_rgb)
        self.stack = [img_gray] * self.img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(self.repeat_actions):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img = self.preprocess(img_rgb)
        self.stack.pop(0)
        self.stack.append(img)
        assert len(self.stack) == self.img_stack
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def preprocess(observation):
        # crop
        observation = observation[0:84]
        # greyscale
        observation = np.dot(observation[..., :3], [0.2989, 0.5870, 0.1140])
        # downsample
        observation = observation[0::2, 0::2]
        # rescale numbers between 0 and 1
        max_val = observation.max() if observation.max() > abs(observation.min()) else abs(observation.min())
        if max_val != 0:
            observation = observation / max_val
        return observation.astype(np.float32)


    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


env = Env()
agent = Agent(alpha=0.0005, gamma=0.99, n_actions=5, fc1_dims=96,fc2_dims=96)
score_history = []

n_episodes = 100
running_score = 0
state = env.reset()

for i in range(n_episodes):
    score = 0
    observation = env.reset()

    for t in range(1000):
        discrete_action = agent.choose_action(observation)
        continuous_action = continuize_action(discrete_action)
        observation_, reward, done, die = env.step(continuous_action)
        env.render()
        agent.store_transition(observation, discrete_action, reward)
        score += reward
        observation = observation_
        if done or die:
            break
    agent.learn()
    running_score = running_score * 0.99 + score * 0.01
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'running score: %.1f' % running_score)
