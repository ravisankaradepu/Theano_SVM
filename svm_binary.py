import theano.tensor as T
import theano
import numpy as np
import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from logistic_sgd import load_data
from neupy.algorithms.gd.hessian import find_hessian_and_gradient

X = T.dmatrix()
Y = T.vector()
w=theano.shared(value=np.random.rand(1,784))
b = theano.shared(value=np.asarray(1.), name='lk', borrow=False)

y = Y*T.dot(w,X)+b

datasets=load_data('mnist.pkl.gz')
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

cost = T.mean(0.01*(w ** 2).sum()+ T.maximum(0,1-y))

gradient = T.grad(cost=cost, wrt=w)
gb=T.grad(cost=cost,wrt=b)

updates = [[w, w - gradient * 0.001],[b,b-gb*0.001]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
batch_size=50
num_batches=np.shape(train_set_x)[0]/batch_size
for e in range(100):
    print "epoch",e
    for i in range(num_batches):
        train(np.matrix.transpose(train_set_x[50*i:50*i+50]), train_set_y[50*i:50*i+50])
#############################################        
symMatrix = T.dmatrix("symMatrix")
symEigenvalues, eigenvectors = T.nlinalg.eig(symMatrix)
get_Eigen = theano.function([symMatrix], [symEigenvalues, eigenvectors])
#############################################
H= theano.shared(value=numpy.zeros((784,784)),name='H',borrow=True)
hessian,_ = find_hessian_and_gradient(cost,[w])
updates_end = [[w, w - gradient * 0.001],[b,b-gb*0.001],[H,hessian]]
train_end = theano.function(inputs=[X, Y], outputs=cost, updates=updates_end, allow_input_downcast=True)
train_end(np.matrix.transpose(train_set_x), train_set_y)
print H.get_value()
ev,_=get_Eigen(H.get_value())

print("zero ev",numpy.sum(ev==0))
print("pos ev",numpy.sum(ev>0))
print("neg ev", numpy.sum(ev<0))

out=theano.shared(value=np.zeros((1,2055)), name='o',borrow=False)
v = T.dot(w,X)+b
valid = theano.function(inputs=[X], outputs=v, updates=[[out,v]],allow_input_downcast=True)
valid(np.transpose(valid_set_x))
pred=out.get_value()
count=0
for i in range(2055):
    if(pred[0,i] > 0 and valid_set_y[i]==1):
        count=count+1
    elif(pred[0,i]<0 and valid_set_y[i]==-1):
        count=count+1
print "Accuracy is ",count/2055.0
