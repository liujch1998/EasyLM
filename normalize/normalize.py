import csv
import numpy as np

f = open('wandb_export_2024-04-29T00_43_22.822-07_00.csv')
reader = csv.DictReader(f)
rewards = []
for row in reader:
    rewards.append(float(row['reward']))
f.close()

mean = np.mean(rewards)
std = np.std(rewards)
print(mean, std)

gain = 1.0 / std
bias = -mean / std
print(gain, bias)
