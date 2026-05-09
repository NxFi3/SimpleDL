import numpy as np

class Layer:
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, dz):
        raise NotImplementedError
    
    def update(self, lr):
        pass


class Sigmoid(Layer):
    def forward(self, x):
        self.x = x if isinstance(x, np.ndarray) else np.array([x])
        return 1 / (1 + np.exp(-self.x))
    
    def backward(self, dz):
        sig = 1 / (1 + np.exp(-self.x))
        return dz * sig * (1 - sig)

class ReLU(Layer):
    def forward(self, x):
        self.x = x if isinstance(x, np.ndarray) else np.array([x])
        return np.maximum(0, x)
    
    def backward(self, dz):
        return dz * (self.x > 0)

class Tanh(Layer):
    def forward(self, x):
        self.x = x if isinstance(x, np.ndarray) else np.array([x])
        self.output = np.tanh(self.x) 
        return self.output
    
    def backward(self, dz):
        return dz * (1 - self.output ** 2)
class Softmax(Layer):
    def forward(self, x):
        self.x = x
        exp_x = np.exp(x - np.max(x))  
        self.output = exp_x / np.sum(exp_x)
        return self.output
    
    def backward(self, dz):
        s = self.output.reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        return np.dot(jacobian, dz)
    
class Dropout(Layer):
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x, training=True):
        self.training = training
        if not training:
            return x
        
        self.mask = np.random.binomial(1, 1-self.p, size=x.shape)
        return x * self.mask / (1-self.p)
    
    def backward(self, dz):
        if not self.training:
            return dz
        return dz * self.mask / (1-self.p)
