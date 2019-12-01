#!/bin/bash

# Default Setting(Notify on state change: BEGIN, END, FAIL or ALL.)
#SBATCH --mail-type=END
#SBATCH --mail-user=cancan_huang@brown.edu

#SBATCH --account=brubenst-condo

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH -J OC_Swimmer-v2_9
#SBATCH --mem=10G

# RUNNING CPU(skylake>broadwell>haswell>ivy>sandy)
###SBATCH --constraint=skylake

# get the run environment
module list

# run command
python oc.py -e 'Swimmer-v2'
# deal the data file

