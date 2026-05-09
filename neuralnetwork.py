from typing import Any

from activations import Dropout
import numpy as np



class Layer:
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, dz):
        raise NotImplementedError
    
    def update(self, lr):
        pass

class Sequential(Layer):
    def __init__(self, architecture: list):
        super().__init__()
        self.layers = architecture  
    
    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out) 
        return out
    
    def backward(self, dz):
        for layer in reversed(self.layers):
            dz = layer.backward(dz)
        return dz
    def eval(self):
     
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = False
    
    def train(self):
      
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = True
    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

class Linear(Layer):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        

           
        std = np.sqrt(2.0 / inputs)

        self.Weights = np.random.randn(outputs, inputs) * std  
        self.bias = np.zeros((outputs, 1))
        self.inputs = inputs
        self.dw = None
        self.db = None


    def __call__(self,x):
        if isinstance(x, np.ndarray):
            self.x = x.reshape(1, -1)
        else:
            self.x = np.array([x]).reshape(1, -1)
            
        z = np.dot(self.Weights, self.x.T) + self.bias
        return z.reshape(-1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            self.x = x.reshape(1, -1)
        else:
            self.x = np.array([x]).reshape(1, -1)
            
        z = np.dot(self.Weights, self.x.T) + self.bias
        return z.reshape(-1)
    
    def backward(self, dz):
        dz = dz.reshape(-1, 1)
        self.dw = np.dot(dz, self.x)
        self.db = dz
        dx = np.dot(self.Weights.T, dz).reshape(-1)
        return dx
    
    def update(self, lr):
        self.Weights -= lr * self.dw
        self.bias -= lr * self.db
