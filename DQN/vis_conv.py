
import numpy as np
import gym
from keras.models import load_model
from keract import get_activations, display_activations, display_heatmaps
from utils import compress_statespace_CONV2D


#######################################################################################
#
#                                     Visualization
#
#######################################################################################

model = load_model("Models/20201207-1/DQNmodel")

env = gym.make("CarRacing-v0")
state = env.reset()

for i in range(75):
    state, a, b, c = env.step([0, +1, 0])

state = compress_statespace_CONV2D(state)
state = np.expand_dims(state, axis=0)
state = np.expand_dims(state, axis=3)

activations = get_activations(model, state, auto_compile=True)
display_activations(activations,
                    cmap=None,
                    save=False,
                    directory='.',
                    data_format='channels_last',
                    fig_size=(24, 24),
                    reshape_1d_layers=False)
display_heatmaps(activations,
                 state,
                 save=False)

