import os
import subprocess

os.chdir("C:/Users/gooog/Desktop/Bachelor/Code/bachelor/DomainBed")

bashCommand = '''python -m domainbed.scripts.train --data_dir=./../data/ --algorithm LM_CVAE --dataset LM_PACS_Debug --test_env 2 --hidden_layer_sizes "[16,8]" --gpus "0" --output_dir ./../logs/DB_test/'''
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
