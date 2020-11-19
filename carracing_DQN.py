import gym
import numpy as np


env = gym.make('CarRacing-v0')
best_score = 0





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

observation = env.reset()
for _ in range(15):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

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

    compressed_statespace = compressed_statespace_84x84_normalized.flatten()
    # flat the matrix to a one-dimensional vector for the NN to read

    # don't know why elsheik is doing frame*2-1 tbh.... maybe to amplify 'color' intensity?

    return compressed_statespace


#print(observation)
#print(observation.shape)
#compressed_observation = compress_statespace(observation)
#print(compressed_observation)
#print(compressed_observation.shape)






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
    if action == 1: quasi_continuous_action = [-1, 0, 0.0]  # Left
    if action == 2: quasi_continuous_action = [+1, 0, 0.0]  # Right
    if action == 3: quasi_continuous_action = [0, +1, 0.0]  # Accelerate
    if action == 4: quasi_continuous_action = [0, 0, 0.8]  # break


    return quasi_continuous_action


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
from keras.layers import Dense, Input
from keras.optimizers import Adam
from collections import deque
from keras.utils.vis_utils import plot_model



class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85           # discount factor
        self.epsilon = 0.5          # exploration vs explotation, i.e. rate to deviate to random actions # TODO I SET THE VALUE LOW ON PUROPSE TO PROVOKE THE EXPLOIT ERROR (set back to 1 after)
        self.epsilon_min = 0.01     # minimum epsilon
        self.epsilon_decay = 0.995  # how epsilon evolves (we want much exploration at the beginning and only few in the end)
        self.learning_rate = 0.005  # rate to update "Q-table"
        self.tau = .125             # rate to update target (goal) model

        self.model = self.create_model()            # create model (what actions to take)
        self.target_model = self.create_model()     # and target model (and what actions we want it to take) this is done not to vary the goal while training

    def create_model(self):         # initiate model and add layers to NN, define activation function and shapes of in- and output
        model = Sequential()

        model.add(Input(shape=7056))
        model.add(Dense(1000, name = "layer0", activation="relu"))             # TODO Get shape right
        model.add(Dense(100, name = "layer1", activation="relu"))              # TODO choose sensible layers (did low for initial performance)
        model.add(Dense(5, name = "layer2", activation="sigmoid"))             # TODO Get shape right
        print(model.output_shape)
        model.compile(loss="mean_squared_error",optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model




    def act(self, state):                                   # funtion to actually perform actions. input state, returns action

        # NOT COMPRESS AS THE FUNCTION IS ONLY FED WITH COMRESSED

        self.epsilon *= self.epsilon_decay                  # let epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon)  # make sure it's not lower than minimum
        if np.random.random() < self.epsilon:               # randomly decide if exploit or explore
            print("DEBUG explore")
            random_action = random.randint(0,4)
            return random_action                               # explore

        else:
            print("DEBUG exploit")
            print(state)
            print(state.shape)
            print(self.model.predict(state))

            # TODO the model.predict should return for the given state Q-values for each of the 5 actions. However, it returns
            # TODO ValueError: Input 0 of layer sequential is incompatible with the layer: expected axis -1 of input shape to have value 7056 but received input with shape [None, 1]
            # TODO this while the input is obviously shape 7056 as shown by the print

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

        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample     # get a random state from the samples

            target = self.target_model.predict(state)           # predict what to do with target model, given random state
            print("replay",target[0][action])

            if done:
                target[0][action] = reward                      # does it return done? if yes nice
            else:
                Q_future = max(self.target_model.predict(new_state)[0]) # what is the future value of that state
                target[0][action] = reward + Q_future * self.gamma      # adjust "Q-table"-entry with immediate reward, and discounted future Q-value
            self.model.fit(state, target, epochs=1, verbose=0)          # fit model to this "Q-table" (i.e. Q-values to states)












    def target_train(self):                                     # reorient goals, i.e. copy the weights from the main model into the target model
        weights = self.model.get_weights()                      # since this is done less frequently, it doesn't distort goals while training
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau) # adjust target weights at rate tau
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

















def main():
    env = gym.make("CarRacing-v0")

    trials = 1          # aka episodes
    trial_len = 200     # how long one episode is

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        cur_state = compress_statespace(env.reset())         # COMPRESS current state
        for step in range(trial_len):
            if step % 100 == 0:
                print("step:",step)

            action = dqn_agent.act(cur_state)               # act given current state, either explore or exploit
            print("DEBUG main: action by dqn:", action)
            action = transform_action(action)               # TRANSFORM ACTION
            print("DEBUG main: action to step:", action)

            new_state, reward, done, _ = env.step(action)   # actual result of act chosen by dqn_agent.act()

            new_state = compress_statespace(new_state)      # COMPRESS new state

            # reward = reward if not done else -20          # don't know what that has been ???
            # new_state = new_state.reshape(1, 2)           # NOT NEEDED SINCE WE ALREADY COMPRESS???

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



