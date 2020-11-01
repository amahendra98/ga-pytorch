#!/bin/bash
#SBATCH --output out/output04.err                                                   # output log file
#SBATCH -e out/error04.err                                                   # error log file
#SBATCH --mem=40G                                                      # request 40G memory
#SBATCH -c 1                                                           # request 6 gpu cores                                    
#SBATCH -p collinslab                                     # request 1 gpu for this job
#SBATCH --exclude=dcc-collinslab-gpu-[01,02,03]

#module load Anaconda/5.3.1                                            # load conda to make sure use GPU version of tf
# add cuda and cudnn path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/apps/rhel7/cudnn/lib64:$LD_LIBRARY_PATH
# add my library path
export PYTHONPATH=$PYTHONPATH:/hpc/home/amm163/ga-ns/

# execute my file
python hyperswipe_03c.py
# python utils/get_mask.py
# python utils/train_test_split.py
# python plotswipe.py
#python -m cProfile train.py
# python train.py
# python test.py
#python evaluate.py
#python generate_robotic_arm.py
