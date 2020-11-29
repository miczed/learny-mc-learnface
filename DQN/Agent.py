########################################################################################
#                                                                                      #
#                                     DQN                                              #
#                                                                                      #
########################################################################################

import gym
import numpy as np
import random
from NN import DeepQNetwork

env = gym.make('CarRacing-v0')

class Agent(object):

    def __init__(self, gamma, epsilon, lr, batch_size, epsilon_min, epsilon_decay, tau):

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.epsilon_min = epsilon_min

        self.mem_cntr = 0
        self.mem_size = 5000
        self.batch_size = batch_size

        self.lr = lr
        self.model = DeepQNetwork(lr=self.lr)
        self.target_model = DeepQNetwork(lr=self.lr)

        self.state_memory = np.zeros((self.mem_size, 7056), dtype=np.float64)
        self.new_state_memory = np.zeros((self.mem_size, 7056), dtype=np.float64)
        self.action_memory = np.zeros((self.mem_size), dtype=int)
        self.reward_memory = np.zeros((self.mem_size), dtype=np.float64)
        self.terminal_memory = np.zeros((self.mem_size), dtype=np.bool)

    def act(self, state):
        # funtion to actually perform actions. input state, returns action
        # NOT COMPRESS AS THE FUNCTION IS ONLY FED WITH COMPRESSED
        self.epsilon *= self.epsilon_decay                  # let epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon)  # make sure it's not lower than minimum

        if np.random.random() < self.epsilon:               # randomly decide if exploit or explore
            print("DEBUG explore")
            random_action = random.randint(0,4)
            #print(random_action)
            return random_action                               # explore

        else:

            print("DEBUG exploit")
            # exploit : the model predicts for each of the five actions the resulting Q value .
            # we choose the "action" (with argmax) highest Q value
            # should return integer e.g. 2 for "left", will be transformed in main()

            state = np.reshape(state,(1,84,84))    # this function somehow fixes the act() and replay() part, however i have no idea if it's still doing what it should or why this fixes it
            # nolonger used after change to torch
            pred = self.model(predict(state))
            print(pred)


            action = np.argmax(pred)
            print(action)

            return action


    def remember(self, state, action, reward, new_state, done):         # remember previous state, action, reward
        '''
        https://www.youtube.com/watch?v=wc-FxNENg9U
        this function was rewritten following the Machine Learning with Phil Tutorial
        '''
        index = self.mem_cntr % self.mem_size
        if index > self.mem_size: print("MEMORY ERROR")
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done
        self.mem_cntr += 1


    def replay(self):
        '''
        https://www.youtube.com/watch?v=wc-FxNENg9U
        this function was rewritten following the Machine Learning with Phil Tutorial


        LECUTRE:    2 page 39 batch vs oneline
                    2 page 40 MeanSquearedError for Q approximation
                    2 page 42 Cross-entropy error?
                    2 page 44 experience replay
                    2 page 44 they subtract current Q value!

        '''
        batch_size = 32

        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample  # get a random state from the samples

            state = np.reshape(state, (1, 84, 84))  # TODO same as act() ???
            new_state = np.reshape(new_state, (1, 84, 84))  # TODO same as act() ???
            action_num = transform_action_backwards(action)
            # print("DEBUG replay: action, num_action and reward:", action, action_num, reward)
            target = self.target_model.predict(state)  # predict what to do with target model, given random state
            print("Debug replay: target:", target)
            # print("Debug replay: model:", self.model.predict(state))

            if done:
                target[0][action_num] = reward  # does it return done? if yes nice, put in final reward
            else:
                Q_future = max(self.target_model.predict(new_state)[
                                   0])  # what is the future value of that state after the action that was taken
                target[0][
                    action_num] = reward + Q_future * self.gamma  # adjust "Q-table"-entry with immediate reward, and discounted future Q-value for tha action that was taken
                print("Debug replay: new target:", target)
            self.model.fit(state, target, epochs=1,
                           verbose=0)  # fit model to this "Q-table" (i.e. Q-values for each of the five actions to a given state)


def target_train(self):                                     # reorient goals, i.e. copy the weights from the main model into the target model

        weights = self.model.get_weights()
        target_weights = self.target_model
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)


