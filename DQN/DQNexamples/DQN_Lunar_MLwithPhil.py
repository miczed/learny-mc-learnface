
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)       #this does return estimates of PV(Q-value)
        return  actions


class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, epsilon_min):

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.99
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.mem_size = 10000
        self.epsilon_min = epsilon_min

        self.action_space = [i for i in range(self.n_actions)]
        self.mem_cntr = 0
        self.Q_eval = DeepQNetwork(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, fc1_dims=256, fc2_dims=256)




        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transitions(self, state, action, reward, new_state, done):

        # This is the memory function to create samples to learn from

        index = self.mem_cntr % self.mem_size #starts at 0 when reaching mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def act(self, observation):

        print(observation)
        print(observation.shape)
        print(type(observation[0]))

        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size: # skip learning till enough samples
            return

        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]    # values for actions that were took


        #print(self.Q_eval.forward(state_batch))
        #print(batch_index)
        #print(action_batch)
        #print(q_eval)


        q_next = self.Q_eval.forward(new_state_batch)                           # value of new state (TODO TARGET NETWORK HERE)

        #print(q_next)


        q_next[terminal_batch] = 0.0

        #print(q_next)
        #print(reward_batch)
        q_target = reward_batch +self.gamma * T.max(q_next, dim=1)[0]

        #print(q_target)



        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)


import gym


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=3, n_actions=4, epsilon_min=0.01, input_dims=[8], lr=0.003)
    scores = []
    eps_history = []
    episodes = 500

    for episode in range(episodes):

        score = 0
        done = False
        cur_state = env.reset()

        while not done:
            print(cur_state)
            action = agent.act(cur_state)
            new_state, reward, done, info = env.step(action)
            score += reward
            agent.store_transitions(cur_state, action, reward, new_state, done)
            agent.learn()
            cur_state = new_state
            scores.append(score)
            eps_history.append(agent.epsilon)
            avg_score = np.mean(scores[-100:])

            print("Epsiode", episode, "Score %.2f" % score, "Average Score %.2f" % avg_score, "Epsilon %.2f" %agent.epsilon)





