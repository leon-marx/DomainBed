import os
import subprocess

os.chdir("C:/Users/gooog/Desktop/Bachelor/Code/bachelor/DomainBed")

bashCommand = 'python -m domainbed.scripts.train --algorithm BIG_LM_CCVAE --hidden_sizes "[1024,512]" --K 10 --lr 1e-06 --dataset LM_PACS --data_dir ./../data/ --output_dir ./../logs/DB_BIG_CCVAE_1/ --test_env 2 --gpu 0 --steps 100 --checkpoint_freq 10 --save_best_every_checkpoint'
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
