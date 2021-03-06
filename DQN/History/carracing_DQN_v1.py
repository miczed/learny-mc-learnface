import gym
import numpy as np

env = gym.make('CarRacing-v0')

#######################################################################################
#
#                                     RANDOM AGENT
#
#######################################################################################

'''
for i in range(1): #Here state the number of games do you want the AI to play
    average_score=[]
    cnt=0
    done=False
    observation=env.reset()
    while not done:
        cnt+=1
        env.render()
        action = env.action_space.sample() # take a random action
        observation,reward,done,info=env.step(action)
        if done:
            print(reward)
            average_score.append(reward)
            break
    best_score=max(average_score)
env.close()
'''

#######################################################################################
#
#                                     PREPARATIONS
#
#######################################################################################

def compress_statespace(raw_statespace):

    '''

    :param raw_statespace:          The statespace as generated by gym in RGB.
                                    The statespace is an np.array with dimensions 96x96x3,
                                    which is the RGB of every pixel in the output

    :return: compressed_statespace: a compressed statespace in grayscale with
                                    dimension 7056(x1)

    this function
    - cuts away unused pixels
    - converts the state_space to grayscale
    - and normalizes the values to 0 to 1

    Function by https://github.com/elsheikh21/car-racing-ppo
    '''

    statespace_84x84 = raw_statespace[:-12, 6:-6]
    # this cuts away the lowest 12 pixels aswell as 6 left and right. returns 84x84

    compressed_statespace_84x84 = np.dot(statespace_84x84[...,0:3], [0.299, 0.587, 0.114])
    # scalar multiplication (dot-product) of every pixel with these values. these values are given by international
    # standards https://www.itu.int/rec/R-REC-BT.601-7-201103-I/en

    compressed_statespace_84x84_normalized = compressed_statespace_84x84/255.0
    # normalize the gray values to values between 0 and 1 (don't know if necessary)

    return compressed_statespace_84x84_normalized

    #compressed_statespace = compressed_statespace_84x84_normalized.flatten()
    # flat the matrix to a one-dimensional vector for the NN to read
    # don't know why elsheik is doing frame*2-1 tbh.... maybe to amplify 'color' intensity?
    #return compressed_statespace

def transform_action(action):

    '''

    :param action:                      a discretized action_space as a single integer
                                        0 = nothing
                                        1 = hard left
                                        2 = hard right
                                        3 = full accelerating
                                        4 = (mild?) breaking

    :return: quasi_continuous_action:   The action_space as generated by gym [n, n, n]
                                        for [steering, accelerating, breaking]
                                        can be -1 to 1 for steering and 0 to 1 for accelerating and breaking
                                        these are continuous values


    This function is used to transform the actions generated by the NN to a format that the environment can use

    Function by https://github.com/NotAnyMike/gym/blob/master/gym/envs/box2d/car_racing.py

    '''

    if action == 0: quasi_continuous_action = [0, 0, 0.0]  # Nothing
    elif action == 1: quasi_continuous_action = [-1, 0, 0.0]  # Left
    elif action == 2: quasi_continuous_action = [+1, 0, 0.0]  # Right
    elif action == 3: quasi_continuous_action = [0, +1, 0.0]  # Accelerate
    elif action == 4: quasi_continuous_action = [0, 0, 0.5]  # break
    else: print("action faulty for action transform", action)

    return quasi_continuous_action

def transform_action_backwards(quasi_continuous_action):

    '''
    :param: quasi_continuous_action:   The action_space as generated by gym [n, n, n]
                                        for [steering, accelerating, breaking]
                                        can be -1 to 1 for steering and 0 to 1 for accelerating and breaking
                                            hese are continuous values


    :return: action:                    a discretized action_space as a single integer
                                        0 = nothing
                                        1 = hard left
                                        2 = hard right
                                        3 = full accelerating
                                        4 = (mild?) breaking




    This function is used to transform the actions generated by the environment to a format used for indexing the NN output

    '''

    if quasi_continuous_action == [0, 0, 0.0]: action = 0  # Nothing
    elif quasi_continuous_action == [-1, 0, 0.0]: action = 1 # Left
    elif quasi_continuous_action == [+1, 0, 0.0] : action = 2  # Right
    elif quasi_continuous_action == [0, +1, 0.0]: action = 3 # Accelerate
    elif quasi_continuous_action == [0, 0, 0.5]: action = 4  # break
    else: print("action faulty for action transform", quasi_continuous_action)

    return action









# TESTING AND STUFF



observation = env.reset()                                                    # Needs to be left in for the model initiation
#for _ in range(15):
#    env.render()
#    action = env.action_space.sample()
#    observation, reward, done, info = env.step(action)
#print(observation)
#print(observation.shape)
compressed_observation = compress_statespace(observation)                   # Needs to be left in for the model initiation
#print(compressed_observation)
#print("compressed observation space shape", compressed_observation.shape)

#action = 1 # e.g. turn left
#print("before",action)
#action = transform_action(action)
#print("after",action)













#######################################################################################
#
#                                     DQN
#
#######################################################################################

import random
from keras.models import Sequential
from keras.layers import Dense, Input, Reshape
from keras.optimizers import Adam
from collections import deque
from keras.utils.vis_utils import plot_model

class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.8           # discount factor
        self.epsilon = 1            # exploration vs explotation, i.e. rate to deviate to random actions
        self.epsilon_min = 0.01     # minimum epsilon
        self.epsilon_decay = 0.999  # how epsilon evolves (we want much exploration at the beginning and only few in the end)
        self.learning_rate = 0.8  # ???
        self.tau = 1             # rate to update target (goal) model
        #self.tau_decay = 0.999           # own idea: maybe learn faster at the beginning?
        #self.tau_min = 0.1

        self.model = self.create_model()            # create model (what actions to take)
        self.target_model = self.create_model()     # and target model (and what actions we want it to take) this is done not to vary the goal while training.
                                                    # MachineLearningWithPhil calls the two models: next and eval: for

    def create_model(self):         # initiate model and add layers to NN, define activation function and shapes of in- and output
        model = Sequential()


        model.add(Input(shape=(compressed_observation.shape)))
        model.add(Reshape((7056,)))                                 # replaces the flatten from the compress statespace
        model.add(Dense(200, name = "layer1", activation="relu"))
        model.add(Dense(5, name = "layer2", activation="relu"))

        print("Input shape", model.input_shape)
        print("Output shape", model.output_shape)
        model.compile(loss="mse",optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model




    def act(self, state):                                   # funtion to actually perform actions. input state, returns action

        # NOT COMPRESS AS THE FUNCTION IS ONLY FED WITH COMPRESSED

        self.epsilon *= self.epsilon_decay                  # let epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon)  # make sure it's not lower than minimum
        if np.random.random() < self.epsilon:               # randomly decide if exploit or explore
            #print("DEBUG explore")
            random_action = random.randint(0,4)
            return random_action                               # explore

        else:
            #print("DEBUG exploit")
            state = np.reshape(state,(1,84,84)) # TODO this function somehow fixes the act() and replay() part, however i have no idea if it's still doing what it should or why this fixes it

            # exploit : the model predicts for each of the five actions the resulting Q value .
            # we choose the "action" (with argmax) highest Q value
            # should return integer e.g. 2 for "left", will be transformed in main()
            return np.argmax(self.model.predict(state))


    def remember(self, state, action, reward, new_state, done):         # remember previous state, action, reward

               # NOT COMPRESS AS THE FUNCTION IS ONLY FED WITH COMRESSED

        self.memory.append([state, action, reward, new_state, done])




    def replay(self):

        '''

        # THIS FUNCTION WILL WORK FINE ONCE WE MANAGE TO DISCRETIZE THE ACTION SPACE, ELSE IT SHOULD BE REWRITTEN
         # after discretizing, the action is a int like "3" and the indexing will work

        '''

        batch_size = 100

        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample     # get a random state from the samples

            state = np.reshape(state, (1, 84, 84)) # TODO same as act() ???
            new_state = np.reshape(new_state, (1, 84, 84)) # TODO same as act() ???
            action_num = transform_action_backwards(action)
            #print("DEBUG replay: action, num_action and reward:", action, action_num, reward)
            target = self.target_model.predict(state)           # predict what to do with target model, given random state
            print("Debug replay: target:", target)
            #print("Debug replay: model:", self.model.predict(state))

            if done:
                target[0][action_num] = reward                      # does it return done? if yes nice, put in final reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])     # what is the future value of that state after the action that was taken
                target[0][action_num] = reward + Q_future * self.gamma      # adjust "Q-table"-entry with immediate reward, and discounted future Q-value for tha action that was taken
                print("Debug replay: new target:", target)
            self.model.fit(state, target, epochs=1, verbose=0)          # fit model to this "Q-table" (i.e. Q-values for each of the five actions to a given state)


    # TODO this is shit: the replay consists of 4/5 not accelerating. Thus the NN gets trained to not accelerate because positive rewards from accelerating get averaged out by the other
    # TODO in other words: if i feed the NN a sample with steering left, it also feeds the NN a *bad* sample of accelerating










    def target_train(self):                                     # reorient goals, i.e. copy the weights from the main model into the target model

        #self.tau *= self.tau_decay  # let tau decay
        #self.tau = max(self.tau_min, self.tau)  # make sure it's not lower than minimum

        weights = self.model.get_weights()                      # since this is done less frequently, it doesn't distort goals while training
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau) # adjust target weights at rate tau
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.target_model.save(fn)










import time

start = time.time()

TRIAL_ID = "20201126h"


def main():
    env = gym.make("CarRacing-v0")
    env = gym.wrappers.Monitor(env, "{}/recordings".format(TRIAL_ID))

    trials = 1          # aka episodes (original 1000)
    trial_len = 200     # how long one episode is (don't know what makes sense? must be over 900 right?)

    dqn_agent = DQN(env=env)
    step = []
    for trial in range(trials):

        cur_state = compress_statespace(env.reset())         # COMPRESS current state
        score = 0

        for step in range(trial_len):

            if step % 10 == 0: #print every n-th step
                print("trial:", trial+1, "of", trials, "| step:",step+10, "of",trial_len)

            action = dqn_agent.act(cur_state)               # act given current state, either explore or exploit
            #print("\tact: ", action)
            #print("DEBUG main: action by dqn:", action)
            action = transform_action(action)               # TRANSFORM ACTION
            #print("DEBUG main: action to step:", action)

            new_state, reward, done, _ = env.step(action)   # actual result of act chosen by dqn_agent.act()
            new_state = compress_statespace(new_state)      # COMPRESS new state

            score = score + reward

            #print("\tremember")
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            #print("\treplay")
            dqn_agent.replay()                              # internally iterates default (prediction) model
            #print("\ttrain")
            dqn_agent.target_train()                        # iterates target model

            cur_state = new_state

            if done:
                print("done in trial:",trial,". With score:",score)
                break

        if score <= 50:                                                 # after 'for loop' finishes or done, check if score is <900 then print fail         # TODO score >900
            print("Finished trial {}, but only reached {} points".format(trial, round(score,0)))
            dqn_agent.save_model("{}/DQNmodel_{}".format(TRIAL_ID, trial))

        else:                                                           # after 'for loop' finishes or done, check if step >900 then print success
            print("COMPLETED!!! reached {} points at step {} in trial {}".format(score, step, trial))
            dqn_agent.save_model("DQNmodel_SUCCESSFUL")
            break

    plot_model(dqn_agent.model, to_file='{}/DQNmodelNN.png'.format(TRIAL_ID), show_shapes=True, show_layer_names=True)


if __name__ == "__main__":
    main()


end = time.time()
print("Elapsed time:", round((end-start)/60,1)," Minutes")

import os
os.system('play -nq -t alsa synth {} sine {}'.format(0.2, 400))
os.system('play -nq -t alsa synth {} sine {}'.format(0.1, 600))

