
########################################################################################
#                                                                                      #
#                                     DQN                                              #
#                                                                                      #
########################################################################################

import gym
import numpy as np
import random
from collections import deque

from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Dropout
from keras.optimizers import Adam

from utils import compress_statespace, transform_action, transform_action_backwards, plot_learning_curve



class DeepQNetwork:

    '''

    LECTURE 4   page 4: Neural Networks in general and why they are good for RL
                page X: Lecutre shows Kears example, I changed to Pythorch because the averageing out problem?
                page 26: how many layers
                page 34: issues with NN
                page 35: features should be normalized (done in funcion compress() )
                page 39; learning rate
                page 53: overfitting (obviously not a problem rn)
                page 59: early stop (not yet)


                https://keras.io/api/layers/convolution_layers/convolution2d/
                https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
                https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf


    '''

    def __init__(self, gamma, epsilon, lr, epsilon_min, epsilon_decay, tau, batch_size, mem_size):

        self.lr = lr
        self.model = self.create_model()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.epsilon_min = epsilon_min

        self.mem_cntr = 0
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=self.mem_size)

        self.lr = lr
        self.model = self.create_model()
        self.target_model = self.create_model()

        self.state_memory = np.zeros((self.mem_size, 7056), dtype=np.float64)
        self.new_state_memory = np.zeros((self.mem_size, 7056), dtype=np.float64)
        self.action_memory = np.zeros((self.mem_size), dtype=int)
        self.reward_memory = np.zeros((self.mem_size), dtype=np.float64)
        self.terminal_memory = np.zeros((self.mem_size), dtype=np.bool)

    def create_model(self):

        model = Sequential()
        input_shape = (7056,)
        model.add(Input(shape=input_shape))
        model.add(Dense(units=2000, activation="relu"))
        model.add(Dense(units=100, activation="relu"))
        model.add(Dense(units=5))
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return model



    def act(self, state):
        # funtion to actually perform actions. input state, returns action
        # NOT COMPRESS AS THE FUNCTION IS ONLY FED WITH COMPRESSED
        self.epsilon *= self.epsilon_decay                  # let epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon)  # make sure it's not lower than minimum

        if np.random.random() < self.epsilon:               # randomly decide if exploit or explore
            #print("DEBUG explore")
            random_action = random.randint(0,4)
            #print(random_action)
            return random_action                               # explore

        else:
            #print("DEBUG exploit")
            # exploit : the model predicts for each of the five actions the resulting Q value .
            # we choose the "action" (with argmax) highest Q value
            # should return integer e.g. 2 for "left", will be transformed in main()
            state = np.reshape(state,(1,7056))    # this function somehow fixes the act() and replay() part, however i have no idea if it's still doing what it should or why this fixes it
            # nolonger used after change to torch
            pred = self.model.predict(state)
            #print(pred)
            action = np.argmax(pred)
            #print(action)
            return action


    def remember(self, state, num_action, reward, new_state, done):         # remember previous state, action, reward
        '''

        '''

        self.memory.append([state, num_action, reward, new_state, done])




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

        if len(self.memory) < self.batch_size:
            print("not yet")
            return

        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action_num, reward, new_state, done = sample  # get a random state from the samples
            state = np.reshape(state, (1, 7056))  # TODO same as act() ???
            new_state = np.reshape(new_state, (1, 7056))  # TODO same as act() ???
            Q_pred = self.model.predict(state)  # predict what to do with base model, given random state
            Q_true = Q_pred # declare to "true" Q value

            if done:
                Q_pred[0][action_num] = reward  # does it return done? if yes nice, put in final reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])  # what is the future value of that state after the action that was taken
                Q_true[0][action_num] = reward + Q_future * self.gamma - Q_pred[0][action_num] # adjust the "true" Q value with immediate reward, and discounted future Q-value for tha action that was taken
                #print("Debug replay: Q_true:", Q_true)

            self.model.fit(state, Q_true, epochs=1, verbose=0)

    def target_train(self):                                     # reorient goals, i.e. copy the weights from the main model into the target model

        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)


    def save_model(self, name):
        self.target_model.save(name)




























import time
start = time.time()
TRIAL_ID = "20201129b"


def main():
    env = gym.make("CarRacing-v0")
    env = gym.wrappers.Monitor(env, "Models/{}/recordings".format(TRIAL_ID), force=True)

    agent = DeepQNetwork(gamma=0.8,
                         epsilon=1,
                         epsilon_decay=0.995,
                         batch_size=32,
                         mem_size= 6400,
                         epsilon_min=0.05,
                         tau=0.25,
                         lr=0.01)

    trials = 3          # aka episodes (original 1000)
    trial_len = 50     # how long one episode is (don't know what makes sense? must be over 900 right?)

    step = []
    score_hist = []
    eps_hist = []
    trial_array = []

    for trial in range(trials):

        cur_state = compress_statespace(env.reset())         # COMPRESS current state
        score = 0
        tiles = 0

        for step in range(trial_len):

            if step % 100 == 0: #print every n-th step
                print("\ttrial:", trial, "of", trials-1, "| step:",step, "of",trial_len-100)

            num_action = agent.act(cur_state)               # act given current state, either explore or exploit
            #print("\tact: ", num_action)
            #print("DEBUG main: action by dqn:", num_action)
            action = transform_action(num_action)               # TRANSFORM ACTION
            #print("DEBUG main: action to step:", action)

            new_state, reward, done, _ = env.step(action)   # actual result of act chosen by dqn_agent.act()
            new_state = compress_statespace(new_state)      # COMPRESS new state

            score += reward
            if reward >= 0:
                tiles += 1

            #print("\tremember")
            agent.remember(cur_state, num_action, reward, new_state, done)
            #print("\treplay")
                                        # internally iterates default (prediction) model
            agent.replay()
            #print("\ttrain")


            agent.target_train()

            cur_state = new_state

            if done:
                print("done in trial:",trial,". With score:",round(score,0))
                env.stats_recorder.save_complete()
                break



        score_hist.append(score)
        trial_array.append(trial)
        eps_hist.append(agent.epsilon)

        plot_learning_curve(x=trial_array, scores=score_hist, epsilons=eps_hist, filename="Models/{}/{}".format(TRIAL_ID, "Performance"))

        if score <= 300:                                                 # after 'for loop' finishes or done, check if score is <900 then print fail         # TODO score >900
            print("Finished trial {}, but only reached {} points ({} tiles)".format(trial, round(score,0), tiles))
            env.stats_recorder.save_complete()
            env.stats_recorder.done = True

        else:                                                           # after 'for loop' finishes or done, check if step >900 then print success
            print("COMPLETED!!! reached {} points at step {} in trial {}".format(score, step, trial))
            agent.save_model("Models/{}/DQNmodel_SUCCESSFUL".format(TRIAL_ID))
            env.stats_recorder.save_complete()
            env.stats_recorder.done = True
            break

        agent.save_model("Models/{}/DQNmodel".format(TRIAL_ID))



if __name__ == "__main__":
    main()






end = time.time()
print("Elapsed time:", round((end-start)/60,1)," Minutes")







