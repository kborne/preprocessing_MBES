#!/bin/bash

source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh #this sets environment for psana2. note psana2 not compatible with psana(1)

source /cds/group/pcds/dist/pds/tmo/scripts/setup_env.sh #This sets the latest environment (2022-04-14)

base_path=/reg/d/psdm/tmo/tmoly7820/results/kurtis/preproc
script=$base_path/config_files_test.py
log=/reg/d/psdm/tmo/tmoly7820/results/kurtis/h5_file/logs/run$1_v1_KB.log

if [ -z "$2" ]
then
    n_nodes=2
else
    n_nodes=$2
fi

if [ -z "$3" ]
then
    tasks_per_node=12 #I think there are 16 procs per node for the feh queues, 64 for the ffb and 12 for the 'old' psana qs
else
    tasks_per_node=$3
fi

echo $log
echo $script


sbatch -p psdebugq -N $n_nodes -n $tasks_per_node --output $log --wrap="mpirun python $script $1"
# sbatch -p psanaq -N $n_nodes -n $tasks_per_node --output $log --wrap="mpirun python $script"
# sbatch -p psanaq --output $log --wrap="python $script"