#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Cancan Huang @ 2019-11-20

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import random
import datetime

def datetime_random_str():
    """
    datetime string with two extra digits for image name YYYMMDDHHMMSS-NN
    """
    s = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return s + "-{:04d}".format(random.randint(1,9999))

def load_results(data_path):
    """
    Input:
        - data_path: location of log files. It can be a directory or a specific file.
    Output:
        - data: an array containing data directly loaded from log fiels
        - files: name of the files. if your input is a specific file, it will return the name of agent
        - game: the mujoco environment name
    """
    if os.path.isdir(data_path):
        game = data_path.split('/')[-1]
        files = os.listdir(data_path)
        data = []
        for _file in files:
            if not os.path.isdir(_file):
                #print(_file)
                data.append(np.loadtxt(data_path+'/'+str(_file)))
        return np.array(data), files, game
    elif os.path.isfile(data_path):
        data_path = data_path.split('/')
        # :-24 is to remove the previous datetime string
        return np.loadtxt(data_path), data_path[-1][:-24], data_path[-2]
    else:
        print('Please enter a valid data path.')
        exit()

def plot_mean_standard_error(data, agent):
    """
    Input: 
        - data: 3d array. [# of training times, output steps, output reward]
        - agent: 1d array. string. 
    Output:
        N/A
    """
    print(data.shape)
    x = data[0,:,0]
    e_x = (np.std(data, axis=0) / np.sqrt(data.shape[0]))[:,1]
    m_x = np.mean(data, axis=0)[:,1]
    plt.plot(x, m_x, label=agent)
    plt.fill_between(x, m_x + e_x, m_x - e_x, alpha=0.3)

def plot(data, filenames, game):
    """
    This function is used for plotting different agent performance under the same
    environment.

    Input:
    - data: a 2d array(when path is file) or 3d array(when path is folder) of 
            cumulative rewards, training log files
    - filenames: an 1d array of string, agent trained under different environments
    Output: results will be saved as figures seperated for every environments.
    """
    if isinstance(filenames, str):
        plt.plot(data[:,0], data[:,1], label=filenames)
        ax = plt.gca()
        ax.xaxis.get_major_formatter().set_powerlimits((0,1))
        plt.xlabel('# of steps')
        plt.ylabel('Cumulative reward')
        plt.title('{}'.format(game))
        plt.legend()
        figname = '{}_{}.png'.format(game, datetime_random_str())
        plt.savefig('images/{}'.format(figname))
        print('Images saved as {}'.format(figname))
        plt.close()
    else:
        print('{} results have been found.'.format(len(filenames)))
        agents = []
        for i in range(len(filenames)):
            agents.append(filenames[i][:-24])
        uni_agents = np.unique(np.array(agents))
        print('Game result from agent: {}'.format(uni_agents))
        fig = plt.figure()
        for agent in uni_agents:
            i_data = []
            for i in range(len(agents)):
                if agent == agents[i]:
                    i_data.append(data[i])
            i_data = np.array(i_data)
            plot_mean_standard_error(i_data, agent)
        plt.legend()
        ax = plt.gca()
        ax.xaxis.get_major_formatter().set_powerlimits((0,1))
        plt.xlabel('# of steps')
        plt.ylabel('Cumulative reward')
        plt.title('{}'.format(game))
        figname = '{}_{}.png'.format(game, datetime_random_str())
        plt.savefig('images/{}'.format(figname))
        plt.close()
        print('Images saved as {}'.format(figname))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Plot the agent training results. Please make sure your log file name is following the format of ppo.py, i.e. '%Y%m%d%H%M%S%f'")
    parser.add_argument('-p', '--path_result', type=str, help='path of the results. It can be a directory or a specific file.', default='./data')
    args = parser.parse_args()
    try:
        os.mkdir('./images')
    except OSError as error:
        print(error)
    data, filenames, game = load_results(data_path=args.path_result)
    print(filenames, game)
    plot(data, filenames, game)
