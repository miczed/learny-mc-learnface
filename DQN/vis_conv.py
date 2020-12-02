
import numpy as np
import gym
from keras.models import load_model
from keract import get_activations, display_activations
from utils import compress_statespace_light


#######################################################################################
#
#                                     Visualization
#
#######################################################################################

model = load_model("Models/20201201/DQNmodel")

env = gym.make("CarRacing-v0")
state = env.reset()

for i in range(50):
    state, a, b, c = env.step([0, +1, 0])

state = compress_statespace_light(state)
state = np.expand_dims(state, axis=0)
state = np.expand_dims(state, axis=3)

activations = get_activations(model, state, auto_compile=True)
display_activations(activations, cmap=None, save=False, directory='.', data_format='channels_last', fig_size=(24, 24), reshape_1d_layers=False)

