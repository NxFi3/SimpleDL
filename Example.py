#import module

from SimpleDL import activations ,losses
from SimpleDL import neuralnetwork as nn
from SimpleDL import optimizer as optim
from SimpleDL import utils
import numpy as np

#create Simple Model and Optimizer

model = nn.Sequential(nn.Linear(2,16),activations.ReLU(),nn.Linear(16,2))
optimizer = optim.Adam(model,losses.CrossEntropyLoss(),learning_rate=0.003)

#Create XOR dataset

#encoder for CrossEntropyLoss
encoder = utils.OneHotEncoding(2)

X = np.array([[1,0],[0,1],[1,1],[0,0]])

Y = encoder.encode(np.array([1,1,0,0]))

#Train Model

epoches=100
for ep in range(epoches):
    for x ,y in zip(X,Y):
        pred = model(x)
        loss = optimizer.step(pred,y)
    if ep%10==0:
        print(loss)

# Test Result
for x in X:
    pred = model(x)
    print(f"Input : {x} ,Model Predict : {np.argmax(pred)}")