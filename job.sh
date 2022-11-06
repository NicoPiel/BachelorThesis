#!/bin/bash
#SBATCH --partition=gpu          # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --mem=32G                # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=8:00:00           # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR

module load cuda/11.6
mamba init
mamba activate mmpyg

jupyter nbconvert --to script training_gcn.ipynb
srun python3 training_gcn.py