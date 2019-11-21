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
    datatime string with four extra digits for image name YYYMMDDHHMMSS
    """
    s = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return s + "-{:04d}".format(random.randint(1,9999))

def load_results(data_path):
    if os.path.isdir(data_path):
        files = os.listdir(data_path)
        data = []
        for _file in files:
            if not os.path.isdir(_file):
                data.append(np.loadtxt(data_path+str(_file)))
        return np.array(data), files
    elif os.path.isfile(data_path):
        return np.loadtxt(data_path), data_path.split('/')[-1][:-4]
    else:
        print('Please enter a valid data path.')
        exit()

def plot(data, labels):
    """
    TODO: need to seperate the Environment w.r.t the results
    """
    if isinstance(labels, str):
        print('Only 1 result is plotted.')
        plt.plot(data[:,0], data[:,1], label=labels)
    else:
        for i in range(len(labels)):
            print('{} results have been found.'.format(len(labels)))
            plt.plot(data[i,:,0], data[i,:,1], label=labels[i][:-4])
    ax = plt.gca()
    ax.xaxis.get_major_formatter().set_powerlimits((0,1))
    plt.legend()
    plt.xlabel('# of Steps')
    plt.ylabel('Cumulative reward')
    plt.savefig('images/{}.png'.format(datetime_random_str()))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot the agent training results.')
    parser.add_argument('-p', '--path_result', type=str, help='path of the results. It can be a directory or a specific file.', default='./data')
    args = parser.parse_args()
    try:
        os.mkdir('./images')
    except OSError as error:
        print(error)
    data, labels = load_results(data_path=args.path_result)
    plot(data, labels)
