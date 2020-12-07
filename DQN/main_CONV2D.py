
########################################################################################
#                                                                                      #
#                                     DQN                                              #
#                                                                                      #
########################################################################################

import time
import gym
import pandas as pd
import numpy as np
import random
import pickle
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model

from utils import compress_statespace_CONV2D, transform_action, plot_learning_curve


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

                https://www.youtube.com/watch?v=WvoLTXIjBYU
                https://keras.io/api/layers/convolution_layers/convolution2d/
                https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
                https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
                https://www.youtube.com/watch?v=YRhxdVk_sIs&feature=emb_logo
                https://arxiv.org/pdf/2010.00717.pdf
    '''

    def __init__(self, gamma, epsilon, lr, epsilon_min, epsilon_decay, tau, batch_size, mem_size, reload=False, reload_path=None):

        self.mem_cntr = 0
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=self.mem_size)

        self.lr = lr
        self.reload = reload
        self.reload_path = reload_path
        if self.reload == True:
            self.model = self.load_model(self.reload_path+"DQNmodel")
            self.target_model = self.load_model(self.reload_path+"DQNmodel")

            with open(reload_path+"mem.file", 'rb') as file:
                    mem = pickle.load(file)
                    for i in range(len(mem)):
                        mem_ = mem[i]
                        self.memory.append(mem_)

        else:
            self.model = self.create_model()
            self.target_model = self.create_model()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau



    def load_model(self, path):

        model = load_model(path)
        return model

    def create_model(self):

        model = Sequential()
        input_shape = (999,42,42,1)
        # Model Architecture by Aldape and Sowell, but only half the filters since we downsized the resolution
        # Dropout by Zhang and Sun
        # this reduces the number of trainable parameters by a LOT!
        model.add(Conv2D(32, (5,5), input_shape=input_shape[1:], activation="relu")) #input shape to ignore batch size
        model.add(Dropout(.2))
        model.add(Conv2D(64, (5,5), activation="relu"))         # note stride is how fast the conv window moves. used by Zhang and Sun but not by Aldape and Sowell
        model.add(Dropout(.2))
        model.add(Conv2D(64, (3,3), activation="relu"))
        model.add(Flatten())
        model.add(Dense(32, activation="softmax"))
        model.add(Dense(5)) # note that we take the argmax in the act() function also acts somewhat like a softmax activation
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return model

    def act(self, state):
        '''
        LECTURE:    2 page 41: Explore or exploit
        '''

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

            state = np.expand_dims(state, axis=0)  # TODO WHAT THE HELL WHY IS THIS WORKING
            state = np.expand_dims(state, axis=3)

            # this function somehow fixes the act() and replay() part, however i have no idea if it's still doing what it should or why this fixes it

            pred = self.model.predict(state)
            #print(pred)
            action = np.argmax(pred)
            #print(action)
            return action

    def remember(self, state, num_action, reward, new_state, done):
        # remember previous state, action, reward
        self.memory.append([state, num_action, reward, new_state, done])

    def replay(self):
        '''
        LECUTRE:    2 page 39 batch vs oneline
                    2 page 40 MeanSquearedError for Q approximation
                    2 page 42 Cross-entropy error?
                    2 page 44 experience replay
        '''

        if len(self.memory) < self.batch_size:
            #print("not yet")
            return

        samples = random.sample(self.memory, self.batch_size)

        for sample in samples:
            state, action_num, reward, new_state, done = sample  # get a random state from the samples
            state = np.expand_dims(state, axis=0) # TODO WHAT THE HELL WHY IS THIS WORKING
            state = np.expand_dims(state, axis=3)
            new_state = np.expand_dims(new_state, axis=0)
            new_state = np.expand_dims(new_state, axis=3)



            Q_pred = self.model.predict(state)  # predict what to do with base model, given random state
            Q_true = Q_pred # declare to "true" Q value
            if done:
                Q_pred[0][action_num] = reward  # does it return done? if yes nice, put in final reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])  # what is the future value of that state after the action that was taken (given one takes the best action next)
                Q_true[0][action_num] = reward + Q_future * self.gamma - Q_pred[0][action_num] # adjust the "true" Q value with immediate reward, and discounted future Q-value for tha action that was taken
                #print("Debug replay: Q_true:", Q_true)

            history_callback = self.model.fit(state, Q_true, epochs=1, verbose=0)
            return history_callback

    def target_train(self):                                     # reorient goals, i.e. copy the weights from the main model into the target model

        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_mem(self, name):

        mem = self.memory
        with open(name, 'wb') as file:
            pickle.dump(mem, file, pickle.HIGHEST_PROTOCOL)

    def save_model(self, name):

        self.target_model.save(name)

    def print_summary(self):

        print(self.model.summary())









def main():
    env = gym.make("CarRacing-v0")
    env = gym.wrappers.Monitor(env, "Models/{}/recordings".format(TRIAL_ID), force=True, video_callable=lambda episode_id:True)

    agent = DeepQNetwork(tau=0.5,
                         lr=0.0005,                   # 0.01 by Aldape and Sowell     0.00001 by Zhang and Sun
                         gamma=0.98,
                         epsilon=0.3,
                         epsilon_decay=0.99995,
                         epsilon_min=0.3,          # 0.1 by Aldape and Sowell
                         batch_size=64,             # 64 by Zhang and Sun
                         mem_size= 10000,
                         reload=False,
                         reload_path="Models/20201206-5/")

    trials = 100         # aka episodes              100 by
    trial_len = 1000     # how long one episode is
    kill_score = -99.0           # which score kills episode
    seed = None # or None

    agent.print_summary()
    plot_model(agent.target_model, to_file="Models/{}/model.png".format(TRIAL_ID))  # Plot architecture
    model_data = pd.DataFrame({'tau': [agent.tau],                                  # save training params
                               'lr': [agent.lr],
                               'gamma': [agent.gamma],
                               'eps':[agent.epsilon],
                               'eps_decay':[agent.epsilon_decay],
                               'eps_min':[agent.epsilon_min],
                               'batch_size':[agent.batch_size],
                               'mem_size':[agent.mem_size],
                               'trials':[trials],
                               'steps':[trial_len],
                               'kill_score':[kill_score],
                               'seed':[seed]})
    model_data.to_csv(path_or_buf="Models/{}/{}".format(TRIAL_ID, "model_data"), index=False, float_format="%.3f")

    step = []
    loss_hist = []
    score_hist = []
    eps_hist = []
    trial_array = []

    for trial in range(trials):

        start_trial = time.time()
        if seed is not None:
            env.seed(seed)
        cur_state = env.reset()
        for i in range(50):
            cur_state, a, b, c = env.step([0, 0, 0])    # Wait for the zoom to settle in

        cur_state = compress_statespace_CONV2D(cur_state)         # COMPRESS current state
        score = 0
        tiles = 0
        loss_hist_ = []

        for step in range(trial_len):

            if step % 100 == 0: #print every n-th step
                print("\tTrial:", trial, "of", trials-1, "| Step:",step, "of",trial_len-100)

            num_action = agent.act(cur_state)               # act given current state, either explore or exploit
            #print("\tact: ", num_action)
            #print("DEBUG main: action by dqn:", num_action)
            action = transform_action(num_action)               # TRANSFORM ACTION
            #print("DEBUG main: action to step:", action)
            new_state, reward, done, _ = env.step(action)   # actual result of act chosen by dqn_agent.act()
            new_state = compress_statespace_CONV2D(new_state)      # COMPRESS new state
            score += reward
            if reward >= 0:
                tiles += 1
            #print("\tremember")
            agent.remember(cur_state, num_action, reward, new_state, done)
            #print("\treplay")
            # internally iterates default (prediction) model
            history_callback = agent.replay()

            if history_callback is not None:
                loss_hist_.append(history_callback.history["loss"])
                history_callback = None # to save memory

            #print("\ttrain")
            agent.target_train()
            cur_state = new_state

            if score<kill_score:
                print("\tTrial:", trial, "of", trials-1, "| kill")
                break
            if done:
                print("\tTrial:", trial, "of", trials-1, "| done")
                break

        # Maybe clear here trial specific variables like cur_state and history_callback to save memory?
        cur_state = None

        score_hist.append(score)
        trial_array.append(trial)
        eps_hist.append(agent.epsilon)
        loss_hist.append(np.mean(loss_hist_))
        plot_learning_curve(x=trial_array, scores=score_hist, epsilons=eps_hist, filename="Models/{}/{}".format(TRIAL_ID, "Performance"))
        plot_data = pd.DataFrame({'trial':trial_array, 'score_history':score_hist, 'epsilon_history':eps_hist, 'loss_history':loss_hist})
        plot_data.to_csv(path_or_buf="Models/{}/{}".format(TRIAL_ID,"plot_data"),index=False, float_format="%.3f")
        plot_data = None # to save memory?

        end_trial = time.time()
        time_trial = round((end_trial - start_trial)/60,1)

        if score < 900:                                                 # after 'for loop' finishes or done, check if score is <900 then print fail
            print("Finished trial {} in {} Minutes, but only reached {} points ({} tiles)".format(trial, time_trial, round(score,0), tiles))
            env.stats_recorder.save_complete()
            env.stats_recorder.done = True

        else:                                                           # after 'for loop' finishes or done, check if step >900 then print success
            print("COMPLETED!!! reached {} points at step {} in trial {} after {} Minutes".format(score, step, trial, time_trial))
            agent.save_model("Models/{}/DQNmodel_SUCCESSFUL".format(TRIAL_ID))
            env.stats_recorder.save_complete()
            env.stats_recorder.done = True
            break

        agent.save_model("Models/{}/DQNmodel".format(TRIAL_ID))
        agent.save_mem("Models/{}/mem.file".format(TRIAL_ID))


start = time.time()
TRIAL_ID = "20201206-6"

if __name__ == "__main__":
    main()

end = time.time()
print("Elapsed time:", round((end-start)/60,1)," Minutes")




# TODO Investigate Memory Issue
# Seems like there is no big memory increase after trial is ended with "done" or "kill"