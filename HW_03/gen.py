import numpy as np

rng = np.random

def gen_data(length=50):
	x_seq = np.concatenate([rng.uniform(size=(length, 1)), np.zeros((length, 1))], axis=-1)

	x_seq[rng.randint(length/10), 1] = 1
	x_seq[rng.randint(length/2, length), 1] = 1

	y_hat = np.sum(x_seq[:, 0] *  x_seq[:, 1])
	return x_seq, y_hat


if __name__ == '__main__':
	print genData().gen_data()
