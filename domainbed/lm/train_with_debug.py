import os
import subprocess

os.chdir("C:/Users/gooog/Desktop/Bachelor/Code/bachelor/DomainBed")

bashCommand = 'python -m domainbed.scripts.train --algorithm LM_CVAE --hidden_sizes "[1,1]" --K 2 --dataset LM_PACS_Debug --data_dir ./../data/ --output_dir ./../logs/DB_test/ --test_env 2 --gpu 0 --steps 10 --checkpoint_freq 2 --save_model_every_checkpoint'

process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
