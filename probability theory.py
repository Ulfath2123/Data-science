import numpy as np
sample_space=['H','T']
prob_dist=np.array([0.5,0.5])
num_tosses=1000
toss_res=np.random.choice(sample_space,size=num_tosses,p=prob_dist)
num_heads=np.sum(toss_res=='H')
num_tails=np.sum(toss_res=='T')
print(num_heads)
print(num_tails)
prob_heads=num_heads / num_tosses
prob_tails=num_tails / num_tosses
print(prob_tails)
print(prob_heads)