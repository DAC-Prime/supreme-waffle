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
    datatime string with two extra digits for image name YYYMMDDHHMMSS-NN
    """
    s = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return s + "-{:04d}".format(random.randint(1,99))

def load_results(data_path):
    if os.path.isdir(data_path):
        files = os.listdir(data_path)
        data = []
        for _file in files:
            if not os.path.isdir(_file):
                data.append(np.loadtxt(data_path+'/'+str(_file)))
        return np.array(data), files
    elif os.path.isfile(data_path):
        return np.loadtxt(data_path), data_path.split('/')[-1][:-4]
    else:
        print('Please enter a valid data path.')
        exit()

def plot_mean_standard_error(data, agent):
    """
    x: the steps took for learning
    """
    x = np.arange(data.shape[1])
    e_x = (np.std(data, axis=0) / np.sqrt(data.shape[0]))[:,1]
    m_x = np.mean(data, axis=0)[:,1]
    plt.plot(x, m_x[:,1])
    plt.fill_between(x, m_x + e_x, m_x - e_x)

def plot(data, filenames):
    """
    Input:
    - data: a 2d array(when path is file) or 3d array(when path is folder) of 
            cumulative rewards, training log files
    - filenames: an 1d array of string, agent trained under different environments
    Output: results will be saved as figures seperated for every environments.
    """
    if isinstance(filenames, str):
        plt.plot(data[:,0], data[:,1], label=labels)
        ax = plt.gca()
        ax.xaxis.get_major_formatter().set_powerlimits((0,1))
        plt.savefig('images/{}_{}.png'.format(labels[i][:-4], datatime_random_str()))
    else:
        print('{} results have been found.'.format(len(labels)))
        agents = []
        for i in range(len(filenames)):
            agents.append(filenames[i].split('_')[0])
        uni_agents = np.unique(np.array(agents))
        print('Game result from agent: {}'.format(uni_agents))
        fig = plt.figure()
        for agent in uni_agents:
            i_data = []
            for i in range(len(agents)):
                if agent == i:
                    i_data.append(data[i])
            plot_mean_standard_error(i_data, agent)
        ax = plt.gca()
        ax.xaxis.get_major_formatter().set_powerlimits((0,1))
        plt.savefig('images/{}_{}.png'.format(labels[i][:-4], datatime_random_str()))
        plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot the agent training results.')
    parser.add_argument('-p', '--path_result', type=str, help='path of the results. It can be a directory or a specific file.', default='./data')
    args = parser.parse_args()
    try:
        os.mkdir('./images')
    except OSError as error:
        print(error)
    data, filenames = load_results(data_path=args.path_result)
    plot(data, filenames)
