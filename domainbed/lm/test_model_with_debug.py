import os
import subprocess

os.chdir("C:/Users/gooog/Desktop/Bachelor/Code/bachelor/DomainBed")

bashCommand = 'python -m domainbed.lm.test_model --ckpt_path logs/Dl --ckpt_path logs/DB_CCVAE_NC_mini/abest_model.pkl --raw True --mode ae'

process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()