

# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c

import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque
from keras.utils.vis_utils import plot_model



class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85           # discount factor
        self.epsilon = 1.0          # exploration vs explotation, i.e. rate to deviate to random actions
        self.epsilon_min = 0.01     # minimum epsilon
        self.epsilon_decay = 0.995  # how epsilon evolves (we want much exploration at the beginning and only few in the end)
        self.learning_rate = 0.005  # rate to update "Q-table"
        self.tau = .125             # rate to update target (goal) model

        self.model = self.create_model()            # create model (what actions to take)
        self.target_model = self.create_model()     # and target model (and what actions we want it to take) this is done not to vary the goal while training

    def create_model(self):         # initiate model and add layers to NN, define activation function and shapes of in- and output
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))   # Dense
        model.add(Dense(48, activation="relu"))                             # dense_1
        model.add(Dense(24, activation="relu"))                             # dense_2
        model.add(Dense(self.env.action_space.n))                           # dense_3
        model.compile(loss="mean_squared_error",optimizer=Adam(lr=self.learning_rate))
        return model




    def act(self, state):                                   # funtion to actually perform actions
        self.epsilon *= self.epsilon_decay                  # let epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon)  # make sure it's not lower than minimum
        if np.random.random() < self.epsilon:               # randomly decide if exploit or explore
            return self.env.action_space.sample()               # explore
        return np.argmax(self.model.predict(state)[0])          # exploit




    def remember(self, state, action, reward, new_state, done):         # remember previous state, action, reward
        self.memory.append([state, action, reward, new_state, done])




    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample     # get a random state from the samples
            target = self.target_model.predict(state)           # predict what to do with target model, given random state
            if done:
                target[0][action] = reward                      # does it return done? if yes nice
            else:
                Q_future = max(self.target_model.predict(new_state)[0]) # what is the future value of that state
                target[0][action] = reward + Q_future * self.gamma      # adjust "Q-table" with immediate reward, and discounted future Q-value
            self.model.fit(state, target, epochs=1, verbose=0)          # fit model to this "Q-table" (i.e. action to states)






    def target_train(self):                                     # reorient goals, i.e. copy the weights from the main model into the target model
        weights = self.model.get_weights()                      # since this is done less frequently, it doesn't distort goals while training
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau) # adjust target weights at rate tau
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


def main():
    env = gym.make("MountainCar-v0")
    gamma = 0.9
    epsilon = .95

    trials = 1          # aka episodes
    trial_len = 300     # how long one episode is

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        cur_state = env.reset().reshape(1, 2)
        for step in range(trial_len):
            if step % 100 == 0:
                print("step:",step)

            action = dqn_agent.act(cur_state)               # act given current state
            new_state, reward, done, _ = env.step(action)   # result of act

            # reward = reward if not done else -20          # don't know what that has been ???
            new_state = new_state.reshape(1, 2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()                              # internally iterates default (prediction) model
            dqn_agent.target_train()                        # iterates target model

            cur_state = new_state
            if done:
                print("done in trial",trial)
                break
        if step >= 199:                                                 # after 'for loop' finishes or done, check if step >199 then print fail
            print("However, failed to complete in under 200 steps in trial {}".format(trial))
            if step % 10 == 0:                                          # also save every 10th model
                dqn_agent.save_model("trial-{}.model".format(trial))
        else:                                                           # after 'for loop' finishes or done, check if step <=199 then print success
            print("Completed in {} in trial {}".format(step, trial))
            dqn_agent.save_model("success.model")
            break

    print(dqn_agent.model.summary())
    plot_model(dqn_agent.model, to_file='model_plot3.png', show_shapes=True, show_layer_names=True)


if __name__ == "__main__":
    main()