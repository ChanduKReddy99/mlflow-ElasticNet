import os
import numpy as np

alphas= np.linspace(0.1, 0.9, 3)
l1_ratios= np.linspace(0.3, 0.8, 3)

for alpha in alphas:
    for l1_ratio in l1_ratios:
        print(f'logging experiment- alpha: {alpha}, l1_ratio: {l1_ratio}')
        os.system(f'python elasticnet.py -a {alpha} -l1 {l1_ratio}')

        
        
