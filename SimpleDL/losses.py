import numpy as np



class MSELoss:
    def __call__(self, predict, target):
        return 0.5 * np.mean((predict - target)**2)
    
    def gradient(self, predict, target):
        return (predict - target)
class BinaryCrossEntropyLoss:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.p = None
        
    def __call__(self, predict, target):
        self.p = np.clip(predict, self.eps, 1 - self.eps)
        self.target = target
        return -np.mean(target * np.log(self.p) + (1 - target) * np.log(1 - self.p))
    
    def gradient(self, predict=None, target=None):

        if predict is not None:
            p = np.clip(predict, self.eps, 1 - self.eps)
            return p - (target if target is not None else self.target)
        return self.p - self.target

class CategoricalCrossEntropyLoss:

    def __init__(self, eps=1e-8):
        self.eps = eps
        self.softmax_output = None
        
    def __call__(self, predict, target):

        shifted = predict - np.max(predict)
        exp_x = np.exp(shifted)
        self.softmax_output = exp_x / (np.sum(exp_x) + self.eps)
        self.target = target
        return -np.sum(target * np.log(self.softmax_output + self.eps))
    
    def gradient(self, predict=None, target=None):
  
        if predict is not None:
            shifted = predict - np.max(predict)
            exp_x = np.exp(shifted)
            softmax = exp_x / (np.sum(exp_x) + self.eps)
            return softmax - (target if target is not None else self.target)
        return self.softmax_output - self.target

class CrossEntropyLoss:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.last_predict_shape = None
        self.sigmoid_output = None
        self.softmax_output = None
        
    def __call__(self, predict, target):
        self.target = target
        

        if len(predict.shape) == 0 or predict.shape[-1] == 1:
     
            self.sigmoid_output = 1 / (1 + np.exp(-predict))
            self.sigmoid_output = np.clip(self.sigmoid_output, self.eps, 1 - self.eps)
            return -np.mean(target * np.log(self.sigmoid_output) + 
                           (1 - target) * np.log(1 - self.sigmoid_output))
        else:
 
            shifted = predict - np.max(predict)
            exp_x = np.exp(shifted)
            self.softmax_output = exp_x / (np.sum(exp_x) + self.eps)
            return -np.sum(target * np.log(self.softmax_output + self.eps))
    
    def gradient(self, predict=None, target=None):
        if self.sigmoid_output is not None:
            return self.sigmoid_output - self.target
        else:
            return self.softmax_output - self.target