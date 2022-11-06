#!/bin/bash
#SBATCH --partition=hpc         # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --mem=8G                # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=24:00:00           # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR

mamba init
mamba activate mm

jupyter nbconvert --to script preprocessing_gcn.ipynb
srun python3 preprocessing_gcn.py