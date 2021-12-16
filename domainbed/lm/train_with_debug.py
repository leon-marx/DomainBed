import os
import subprocess

os.chdir("C:/Users/gooog/Desktop/Bachelor/Code/bachelor/DomainBed")

bashCommand = '''python -m domainbed.scripts.train --algorithm LM_CVAE --hidden_sizes "[2,2]" --K 25 --dataset LM_PACS_Debug --data_dir ./../data/ --output_dir ./../logs/DB_test/ --test_env 2 --gpus 0 --steps 10000 --checkpoint_freq 2000 --save_model_every_checkpoint'''
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
