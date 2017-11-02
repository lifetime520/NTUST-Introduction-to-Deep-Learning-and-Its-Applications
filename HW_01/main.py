import theano as th
import theano.tensor as ts
import numpy as np

rnd = np.random

x = ts.dvector("x")
y = ts.dvector("y")
w = th.shared(rnd.random(2), name="w")
b = th.shared(rnd.random(1), name="b")

print("Initial model:")
print(w.get_value())
print(b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + ts.exp(ts.dot(-x, w) - b))       # Probability that target = 1
prediction = p_1 > 0.5                          # The prediction thresholded
xent = -y * ts.log(p_1) - (1-y) * ts.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()      # The cost to minimize
gw, gb = ts.grad(cost, [w, b])                  # Compute the gradient of the cost
                                                # w.r.t weight vector w and
                                                # bias term b
                                                # (we shall return to this in a
                                                # following section of this tutorial)
# Compile
train = th.function(
          inputs=[x, y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = th.function(inputs=[x], outputs=prediction)

# Train
for i in range(100):
    pred, err = train([0, 0], [0])
    pred, err = train([0, 1], [0])
    pred, err = train([1, 0], [0])
    pred, err = train([1, 1], [1])

print "Final model:"
print w.get_value()
print b.get_value()

print "target value:0"
print "prediction on [0,0]: %s" % (predict([0, 0]))
print "prediction on [0,1]: %s" % (predict([0, 1]))
print "prediction on [1,0]: %s" % (predict([1, 0]))

print "target value:1"
print "prediction on [1,1]: %s" % (predict([1, 1]))

