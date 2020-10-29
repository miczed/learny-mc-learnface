
'''

        Self Driving Cab
        Q TABLE

'''


import gym
from IPython.display import clear_output
from time import sleep
import numpy as np
import random

env = gym.make("Taxi-v3").env


'''

SETTING UP

'''





q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1	# learning rate (i.e. rate of updating Q table)
gamma = 0.6	# discount factor (i.e. how to value short term vs long term)
epsilon = 0.1	# possiblity of exploring rather than exploiting

# For plotting metrics
all_epochs = []
all_penalties = []





'''

TRAINING

'''






for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 1000 == 0:
        clear_output(wait=True)
        print(f"Trained: {i/1000}%")

print("Training finished.\n")


print(30*"#")
print("OPTIMAL SOLUTION")
print("\n\n")

print(q_table)

print("\n\n")
print(30*"#")

sleep(5)










"""

Evaluate agent's performance after Q-learning


"""

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print("\n\n")
print("EVALUATION")
print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

sleep(10)








'''

ANIMATING THE OPTIMAL Q TABLE IN ACTION

'''

print("\n\n")
print("The optimal solution in action (slowed down):")

sleep(10)


frames = [] # for animation
state = env.reset()
done = False


while not done:
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1




sleep(2)

print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))



def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(1)


print_frames(frames)
