import os
import subprocess

os.chdir("C:/Users/gooog/Desktop/Bachelor/Code/bachelor/DomainBed")

bashCommand = 'python -m domainbed.scripts.train --algorithm LM_CCVAE --lamb 2 --lr 1e-06 --dataset LM_PACS_Debug --data_dir ./../data/ --output_dir ./../logs/testing/ --test_env 0 --gpu 0 --steps 100 --checkpoint_freq 10 --save_best_every_checkpoint --batch_size 2'
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
