#!/bin/bash
#SBATCH --partition=gpu4          # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --mem=64G                # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=8:00:00           # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR

module load cuda/10.2
mamba init
mamba activate mm

jupyter nbconvert --to script training_nn.ipynb
srun python3 training_nn.py