########################################################################################
#                                                                                      #
#                                     DQN                                              #
#                                                                                      #
########################################################################################

from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Dropout
from keras.optimizers import Adam

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

    def __init__(self, lr):

        self.lr = lr
        self.model = self.create_model()


    def create_model(self):

        model = Sequential()
        input_shape = (7056,)
        model.add(Input(shape=input_shape))
        model.add(Dense(units=2000, activation="relu"))
        model.add(Dense(units=100, activation="relu"))
        model.add(Dense(units=5))
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))

        return model


    def save_model(self, name):
        self.model.save(name)



















