import json
import matplotlib.pyplot as plt
import numpy as np

repo_path = "C:/users/gooog/desktop/bachelor/code/bachelor/"
ckpt_path = "logs/DB_CCVAE_3/"

data = []
with open(repo_path + ckpt_path + "results.jsonl") as f:
    for line in f.readlines():
        data.append(json.loads(line))

for key in data[0]:
    if "env" in key:
        vals = []
        x_vals = []
        for d in data:
            vals.append(d[key])
            x_vals.append(d["epoch"])
        plt.plot(x_vals, vals)
        plt.title(f"Key: {key}")
        plt.show()