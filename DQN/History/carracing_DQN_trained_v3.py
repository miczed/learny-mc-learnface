import gym
import numpy as np
import torch as T
from carracing_DQN_v3 import DeepQNetwork
from preparations import transform_action, transform_action_backwards, compress_statespace




#######################################################################################
#
#                                     DQN
#
#######################################################################################




model = "20201128e_part5"
trial = "42"

dqn_agent_trained = DeepQNetwork(lr=0)
dqn_agent_trained.load_state_dict(T.load("Models/{}/DQNmodel_{}".format(model, trial)))

done=False
score = 0


env = gym.make('CarRacing-v0')
env = gym.wrappers.Monitor(env, "Models/20201127{}/Applied/{}".format(model, trial), force=True)
cur_state = compress_statespace(env.reset())

for i in range(1000):

    env.render()

    cur_state = T.tensor([cur_state]).to(dqn_agent_trained.device)
    actions = dqn_agent_trained.forward(cur_state.float())
    action = T.argmax(actions).item()

    action = transform_action(action)

    new_state, reward, done, _ = env.step(action)
    new_state = compress_statespace(new_state)

    score = reward + score
    cur_state = new_state

    if done:
        print("DONE!")
        break

print(score)
env.close()













