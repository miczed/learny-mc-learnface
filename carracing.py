import gym
import numpy as np
from gym import wrappers

env=gym.make('CarRacing-v0')

best_score=0

for i in range(10): #Here state the number of games do you want the AI to play
    average_score=[]
    cnt=0
    done=False
    observation=env.reset()
    while not done:
        cnt+=1
        #env.render()
        action = env.action_space.sample() # take a random action
        observation,reward,done,info=env.step(action)
        if done:
            print(reward)
            average_score.append(reward)
            break
    best_score=max(average_score)
env.close()
