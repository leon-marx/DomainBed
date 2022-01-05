import os
import subprocess

os.chdir("C:/Users/gooog/Desktop/Bachelor/Code/bachelor/DomainBed")

bashCommand = 'python -m domainbed.scripts.train --algorithm LM_CCVAE --K 3 --ckpt_path ./../logs/DB_BIG_CCVAE_1/best_model.pkl --lamb 2 --lr 1e-06 --dataset LM_PACS --data_dir ./../data/ --output_dir ./../logs/testing/ --test_env 0 --gpu 0 --steps 100 --checkpoint_freq 10 --save_best_every_checkpoint'
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
