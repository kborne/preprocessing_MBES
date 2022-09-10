#!/bin/bash

source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh #this sets environment for psana2. note psana2 not compatible with psana(1)
base_path=/reg/d/psdm/tmo/tmoly7820/results/kurtis/preproc
script=$base_path/config_files_v1.py
log=$base_path/logs/output_run$1.out

if [ -z "$2" ]
then
    n_nodes=1
else
    n_nodes=$2
fi

if [ -z "$3" ]
then
    tasks_per_node=1
else
    tasks_per_node=$3
fi

email=kborne91@gmail.com

echo $log
echo $script

job_name=r_$1

sbatch -p psanaq -J $job_name -N $n_nodes -n $tasks_per_node --output $log --mail-user=$email --mail-type=BEGIN,END --wrap="mpirun python $script $1"
