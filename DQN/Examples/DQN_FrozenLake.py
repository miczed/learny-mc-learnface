
# https://www.baeldung.com/cs/reinforcement-learning-neural-network

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

env = gym.make("FrozenLake-v0")

discount_factor = 0.5
eps = 0.5
eps_decay_factor = 0.999
num_episodes = 2

model = Sequential()
model.add(InputLayer(batch_input_shape=(1, env.observation_space.n)))
model.add(Dense(20, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

for i in range(num_episodes):
    state = env.reset()
    eps *= eps_decay_factor
    done = False
    while not done:
        if np.random.random() < eps:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(model.predict(np.identity(env.observation_space.n)[state:state + 1]))

        new_state, reward, done, _ = env.step(action)
        target = reward + discount_factor * np.max(model.predict(np.identity(env.observation_space.n)[new_state:new_state + 1]))
        target_vector = model.predict(np.identity(env.observation_space.n)[state:state + 1])[0]
        target_vector[action] = target
        model.fit(np.identity(env.observation_space.n)[state:state + 1],
                  target_vector.reshape(-1, env.action_space.n),
                  epochs=1,
                  verbose=0)
        state = new_state

    print("Epiosde:\t",i)
    print("Epsilon:\t",round(eps,3))
    print("Target:\t\t",target)

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



