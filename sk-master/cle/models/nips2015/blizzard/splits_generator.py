import numpy as np

LEN_X = 417937
train_prop = 0.8
valid_prop = 0.1
test_prop  = 0.1
train_end = np.int(np.ceil(LEN_X * train_prop))
valid_end = np.int(np.ceil(LEN_X * valid_prop))

random_shuffle_idx = np.random.permutation(LEN_X)
train_idx = random_shuffle_idx[:train_end]
valid_idx = random_shuffle_idx[train_end:train_end+valid_end]
test_idx  = random_shuffle_idx[train_end+valid_end:]


np.savez('splits_indices.npz', train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
