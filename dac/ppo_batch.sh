#!/bin/bash

# Default Setting(Notify on state change: BEGIN, END, FAIL or ALL.)
#SBATCH --mail-type=END
#SBATCH --mail-user=cancan_huang@brown.edu

#SBATCH --account=brubenst-condo

#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH -J PPO_Swimmer-v2_9
#SBATCH --mem=10G

# RUNNING CPU(skylake>broadwell>haswell>ivy>sandy)
###SBATCH --constraint=skylake

# get the run environment
module list

# run command
python ppo.py -e 'Swimmer-v2'
# deal the data file

