########################################################################################
#                                                                                      #
#                                     DQN                                              #
#                                                                                      #
########################################################################################

import gym
import numpy as np
import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import transform_action, transform_action_backwards, compress_statespace, plot_learning_curve


env = gym.make('CarRacing-v0')













class DeepQNetwork(nn.Module):

    '''

    LECTURE 4   page 4: Neural Networks in general and why they are good for RL
                page X: Lecutre shows Kears example, I changed to Pythorch because the averageing out problem?
                page 26: how many layers
                page 34: issues with NN
                page 35: features should be normalized (done in funcion compress() )
                page 39; learning rate
                page 53: overfitting (obviously not a problem rn)
                page 59: early stop (not yet)


    '''

    def __init__(self, lr):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(7056, 700)
        self.fc2 = nn.Linear(700, 50)
        self.fc3 = nn.Linear(50, 5)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)       #this does return estimates of PV(Q-value)
        return actions

    def save_model(self, name):
        T.save(self.state_dict(), name)







class Agent(object):
    def __init__(self, gamma, epsilon, lr, batch_size, epsilon_min, epsilon_decay, tau, reload=None):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.epsilon_min = epsilon_min

        self.mem_cntr = 0
        self.mem_size = 5000
        self.batch_size = batch_size

        self.lr = lr
        self.model = DeepQNetwork(self.lr)
        self.target_model = DeepQNetwork(self.lr)

        if reload is not None:
            self.model.load_state_dict(T.load(reload))  # I DO THIS TO COMBAT THE MEMEORY LEAK ISSUE
            self.target_model.load_state_dict(T.load(reload))  # I DO THIS TO COMBAT THE MEMEORY LEAK ISSUE


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
            #print("DEBUG explore")
            random_action = random.randint(0,4)
            #print(random_action)
            return random_action                               # explore
        else:
            state = T.tensor([state]).to(self.model.device)
            #print("DEBUG exploit")
            #state = np.reshape(state,(1,84,84))    # this function somehow fixes the act() and replay() part, however i have no idea if it's still doing what it should or why this fixes it
                                                    #nolonger used after change to torch
            # exploit : the model predicts for each of the five actions the resulting Q value .
            # we choose the "action" (with argmax) highest Q value
            # should return integer e.g. 2 for "left", will be transformed in main()
            #print(self.model.predict(state))
            actions = self.model.forward(state.float())
            action = T.argmax(actions).item()
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
        if self.mem_cntr < self.batch_size:
            return
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        indices = np.arange(self.batch_size, dtype=np.int32)

        states = T.tensor(self.state_memory[batch]).to(self.model.device)
        actions = T.tensor(self.action_memory[batch]).to(self.model.device)
        rewards = T.tensor(self.reward_memory[batch]).to(self.model.device)
        new_states = T.tensor(self.new_state_memory[batch]).to(self.model.device)
        terminal = T.tensor(self.terminal_memory[batch]).to(self.model.device)

        q_pred = self.model.forward(states.float())[indices, actions]
        q_next = self.target_model.forward(states.float())

        q_next[terminal] = 0.0
        q_target = rewards + self.gamma * T.max(q_next, dim=1)[0] - q_pred #TODO in Lecutre, they subtract the current Q value?
        loss = self.model.loss(q_target.float(), q_pred.float()).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()


    def target_train(self):                                     # reorient goals, i.e. copy the weights from the main model into the target model
        if np.random.random() < self.tau:
            self.target_model.load_state_dict(self.model.state_dict())
























import time

start = time.time()

TRIAL_ID = "20201128g"


'''

Change Trial ID if you want to save model, else it just overwrites. 

If you want to resume training of a saved model, change the agent function to reload="Modelname", instead of reload=None

NOTE: with these settings i got to run it till trial X step X

'''










def main():
    env = gym.make("CarRacing-v0")
    env = gym.wrappers.Monitor(env, "Models/{}/recordings".format(TRIAL_ID), force=True)

    agent = Agent(gamma=0.8,
                  epsilon=1,
                  epsilon_decay=0.999,
                  batch_size=32,
                  epsilon_min=0.05,
                  tau=0.5,
                  lr=0.01,
                  #reload="Models/20201128f_3/DQNmodel_39")
                  reload=None)

    trials = 100          # aka episodes (original 1000)
    trial_len = 1500     # how long one episode is (don't know what makes sense? must be over 900 right?)

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
            #print("\tact: ", action)
            #print("DEBUG main: action by dqn:", action)
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
            agent.model.save_model("Models/{}/DQNmodel_{}".format(TRIAL_ID, trial))
            env.stats_recorder.save_complete()
            env.stats_recorder.done = True

        else:                                                           # after 'for loop' finishes or done, check if step >900 then print success
            print("COMPLETED!!! reached {} points at step {} in trial {}".format(score, step, trial))
            agent.model.save_model("Models/{}/DQNmodel_SUCCESSFUL".format(TRIAL_ID))
            env.stats_recorder.save_complete()
            env.stats_recorder.done = True
            break



if __name__ == "__main__":
    main()






end = time.time()
print("Elapsed time:", round((end-start)/60,1)," Minutes")

import os
#os.system('play -nq -t alsa synth {} sine {}'.format(0.2, 400))
#os.system('play -nq -t alsa synth {} sine {}'.format(0.1, 600))







