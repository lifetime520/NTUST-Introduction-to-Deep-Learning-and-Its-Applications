import theano as th
import theano.tensor as te
import numpy as np
import time

rng = np.random
LEARNING_RATE = 0.15

X = th.shared(value=np.asarray([[0, 1], [1, 0], [0, 0], [1, 1]]), name='X')
y = th.shared(value=np.asarray([[0], [0], [1], [1]]), name='y')

def layer(**kwargs):
    return th.shared(value=np.asarray(rng.uniform(low=0, high=1, size=kwargs['shape'])), name=kwargs['name'])

W1 = layer(shape=(2, 2), name='W1')
W2 = layer(shape=(2, 1), name='W2')
b1 = layer(shape=2, name='b1')
b2 = layer(shape=1, name='b2')

output = te.nnet.sigmoid(te.dot(te.nnet.sigmoid(te.dot(X, W1) + b1), W2) + b2)
cost = te.mean((y - output) ** 2)
testError = output > 0.5
updates = [(W1, W1 - LEARNING_RATE * te.grad(cost, W1)),
           (W2, W2 - LEARNING_RATE * te.grad(cost, W2)),
           (b1, b1 - LEARNING_RATE * te.grad(cost, b1)),
           (b2, b2 - LEARNING_RATE * te.grad(cost, b2))]

train = th.function(inputs=[], outputs=cost, updates=updates)
test = th.function(inputs=[], outputs=testError)

round = 0
error = 1
while (0.01 <= error and round < 100000):
    error = train()
    round += 1

print 'total %d round, mean error: %f' % (round, error)

testResult = test()
print 'input: [0, 1]; target: 0, test: %d' % int(testResult[0][0])
print 'input: [1, 0]; target: 0, test: %d' % int(testResult[1][0])
print 'input: [0, 0]; target: 1, test: %d' % int(testResult[2][0])
print 'input: [1, 1]; target: 1, test: %d' % int(testResult[3][0])

