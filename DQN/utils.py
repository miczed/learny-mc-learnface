
import numpy as np
import matplotlib.pyplot as plt
import csv

#######################################################################################
#
#                                     PREPARATIONS
#
#######################################################################################

def compress_statespace(raw_statespace):

    '''

    :param raw_statespace:          The statespace as generated by gym in RGB.
                                    The statespace is an np.array with dimensions 96x96x3,
                                    which is the RGB of every pixel in the output

    :return: compressed_statespace: a compressed statespace in grayscale with
                                    dimension 7056(x1)

    this function
    - cuts away unused pixels
    - converts the state_space to grayscale
    - and normalizes the values to 0 to 1

    Function by https://github.com/elsheikh21/car-racing-ppo
    '''

    statespace_84x84 = raw_statespace[:-12, 6:-6]
    # this cuts away the lowest 12 pixels aswell as 6 left and right. returns 84x84
    compressed_statespace_84x84 = np.dot(statespace_84x84[...,0:3], [0.299, 0.587, 0.114])
    # scalar multiplication (dot-product) of every pixel with these values. these values are given by international
    # standards https://www.itu.int/rec/R-REC-BT.601-7-201103-I/en
    compressed_statespace_84x84_normalized = compressed_statespace_84x84/255.0
    # normalize the gray values to values between 0 and 1 (don't know if necessary)
    #return compressed_statespace_84x84_normalized
    compressed_statespace = compressed_statespace_84x84_normalized.flatten()
    # flat the matrix to a one-dimensional vector for the NN to read
    # don't know why elsheik is doing frame*2-1 tbh.... maybe to amplify 'color' intensity?
    return compressed_statespace

def compress_statespace_light(raw_statespace):

    '''

    :param raw_statespace:          The statespace as generated by gym in RGB.
                                    The statespace is an np.array with dimensions 96x96x3,
                                    which is the RGB of every pixel in the output

    :return: compressed_statespace: a compressed statespace in grayscale with
                                    dimension 84x84(x1)

    this function
    - cuts away unused pixels
    - converts the state_space to grayscale
    - and normalizes the values to 0 to 1

    Function by https://github.com/elsheikh21/car-racing-ppo
    '''

    statespace_84x84 = raw_statespace[:-12, 6:-6]
    # this cuts away the lowest 12 pixels aswell as 6 left and right. returns 84x84
    compressed_statespace_84x84 = np.dot(statespace_84x84[...,0:3], [0.299, 0.587, 0.114])
    # scalar multiplication (dot-product) of every pixel with these values. these values are given by international
    # standards https://www.itu.int/rec/R-REC-BT.601-7-201103-I/en
    compressed_statespace_84x84_normalized = compressed_statespace_84x84/255.0
    # normalize the gray values to values between 0 and 1 (don't know if necessary)
    return compressed_statespace_84x84_normalized

def transform_action(action):

    '''

    :param action:                      a discretized action_space as a single integer
                                        0 = nothing
                                        1 = hard left
                                        2 = hard right
                                        3 = full accelerating
                                        4 = (mild?) breaking

    :return: quasi_continuous_action:   The action_space as generated by gym [n, n, n]
                                        for [steering, accelerating, breaking]
                                        can be -1 to 1 for steering and 0 to 1 for accelerating and breaking
                                        these are continuous values


    This function is used to transform the actions generated by the NN to a format that the environment can use

    Function by https://github.com/NotAnyMike/gym/blob/master/gym/envs/box2d/car_racing.py

    '''

    if action == 0: quasi_continuous_action = [0, 0, 0.0]  # Nothing
    elif action == 1: quasi_continuous_action = [-1, 0, 0.0]  # Left
    elif action == 2: quasi_continuous_action = [+1, 0, 0.0]  # Right
    elif action == 3: quasi_continuous_action = [0, +1, 0.0]  # Accelerate
    elif action == 4: quasi_continuous_action = [0, 0, +1]  # break
    else: print("action faulty for action transform", action)

    return quasi_continuous_action

def transform_action_backwards(quasi_continuous_action):

    '''
    :param: quasi_continuous_action:   The action_space as generated by gym [n, n, n]
                                        for [steering, accelerating, breaking]
                                        can be -1 to 1 for steering and 0 to 1 for accelerating and breaking
                                            hese are continuous values


    :return: action:                    a discretized action_space as a single integer
                                        0 = nothing
                                        1 = hard left
                                        2 = hard right
                                        3 = full accelerating
                                        4 = (mild?) breaking




    This function is used to transform the actions generated by the environment to a format used for indexing the NN output

    '''

    if quasi_continuous_action == [0, 0, 0.0]: action = 0  # Nothing
    elif quasi_continuous_action == [-1, 0, 0.0]: action = 1 # Left
    elif quasi_continuous_action == [+1, 0, 0.0] : action = 2  # Right
    elif quasi_continuous_action == [0, +1, 0.0]: action = 3 # Accelerate
    elif quasi_continuous_action == [0, 0, +1]: action = 4  # break
    else: print("action faulty for action transform", quasi_continuous_action)

    return action







def plot_learning_curve(x, scores, epsilons, filename, lines=None, reload=None):

    '''

    Stolen form ML with Phil https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/blob/master/utils.py


    :param x:
    :param scores:
    :param epsilons:
    :param filename:
    :param lines:
    :return:
    '''


    if reload is not None:
        plot_data = csv.reader("plot_data.csv", delimiter=";")
        # TODO finish funtion to plot graphs when resuming training

    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, scores, color="C1")
    ax2.plot(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score and MA20', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
    plt.close('all')




