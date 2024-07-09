import data_utils
from copy import deepcopy
import random
import numpy as np

n_start_offsets = 0
n_samples_per_start = 10000

fn = 'testdata_10k.dill'

#start_offset_variance = 2*[0.1]+[0]+2*[0.0]+[1.5] # variate x,y, yaw for start offset
start_offset_variance = 2*[0.01]+[0]+2*[0.0]+[0.2] # variate x,y, yaw for start offset

start_offsets = []

# have one identity transform as no-offset from start
start_offsets.append(data_utils.Transform())

for i in range(n_start_offsets):
    _, t = data_utils.sample_xyz_rpy_zero_mean(start_offset_variance)
    start_offsets.append(t)



target_from_start_offset = data_utils.Transform()
target_from_start_offset.pos[1] =-0.1

#variance_for_variance = 3*[0.1]+2*[0.4]+[1.5]
sigma = 3*[0.01]+3*[0.03]  # 1cm / ~1.8 deg
variance_for_variance = np.power(sigma,2)
#variance_for_variance = 6*[0]

bool_sample_new_variance_each_time = True

variance = deepcopy(variance_for_variance)

test_entries = []

for so in start_offsets:
    batch = data_utils.create_samples_zero_mean(sample_variance=variance_for_variance, count=n_samples_per_start, sample_variance_each_time=bool_sample_new_variance_each_time)
    for e in batch:
        e.start_offset_is = so
        e.goal_offset_is = target_from_start_offset
    test_entries.extend(batch)

for e in test_entries:
    print(e.sampled_offset, e.sampled_variance)



random.shuffle(test_entries)

data_utils.write_data(test_entries,fn)

#tmp2 = data_utils.load_data("testdata.dill")

#print(f"Loaded {len(tmp2)} entries")





