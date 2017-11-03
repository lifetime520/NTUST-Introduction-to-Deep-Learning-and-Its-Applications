import numpy as np
import theano as th
import theano.tensor as te
import gen_data as gen

# var def area
rng = np.random
n_h, n_i, n_o = 1, 2, 1
learning_rate = 0.35

# train data
[x_init, y_init] = gen.genData().gen_data(50)

def layer(**kwargs):
	low, high = 0, 1
	if 'low' in kwargs:
		low = kwargs['low']
	if 'high' in kwargs:
		high = kwargs['high']
	if 'val' in kwargs:
		return th.shared(value=kwargs['val'], name=kwargs['name'])
	else:
		return th.shared(value=np.asarray(rng.uniform(low=low, high=high, size=kwargs['shape'])), name=kwargs['name'])

# input, target
x_seq, y_hat = layer(val=x_init, name='x_seq'), layer(val=y_init, name='y_hat')

h0 = te.vector('h0')
W_h = layer(shape=(n_h, n_h), low=-.5, high=.5, name='W_h')
W_i = layer(shape=(n_i, n_h), low=-.2, high=.2, name='W_i')
W_o = layer(shape=(n_h, n_o), low=-.9, high=.9, name='W_o')
b_h = layer(shape=(n_h, ), name='b_h')
b_o = layer(shape=(n_o, ), name='b_o')


def step(x_t, a_tm1):
    a_t = te.nnet.sigmoid(te.dot(x_t, W_i) + te.dot(a_tm1, W_h) + b_h)
    y_t = te.nnet.sigmoid(te.dot(a_t, W_o) + b_o)
    return a_t, y_t

[h, y], _ = th.scan(step,
                    sequences=x_seq,
                    outputs_info=[h0, None])
error = te.sum(((y - y_hat) ** 2)) #.sum()
gW_h, gW_i, gW_o, gb_h, gb_o = te.grad(error, [W_h, W_i, W_o, b_h, b_o])
train = th.function(inputs=[h0],
                    outputs=error,
                    updates=[(W_h, W_h - learning_rate * gW_h),
                             (W_i, W_i - learning_rate * gW_i),
                             (W_o, W_o - learning_rate * gW_o),
                             (b_h, b_h - learning_rate * gb_h),
                             (b_o, b_o - learning_rate * gb_o)])

round, error = 0, 1
while (0.01 <= error and round < 1000000):
	error = train([0])
	round += 1
	if round % 100000 == 0:
		print round, error

print 'training finsh. error:%5.3f, round:%d' % (error, round)


x_test = te.dmatrix('x_test')
h_t = te.vector('h_t')
[h, y_t], _ = th.scan(step,
                    sequences=x_test,
                    outputs_info=[h_t, None])
out_test = (y_t).sum()
test = th.function(inputs=[h_t, x_test],
                    outputs=out_test)

for i in range(10):
	[x_gen , y_gen] = gen.genData().gen_data(rng.randint(50, 55))
	guess = test([0], x_gen)
	print 'test: %5.2f, correct:%5.2f' % (guess, y_gen)
