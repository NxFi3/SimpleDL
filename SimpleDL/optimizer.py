import numpy as np
class OptimizerBase:

    def __init__(self, model, loss_fn, learning_rate):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = learning_rate
        self.total_loss = 0
    
    def compute_gradients(self, pred, target):
        loss = self.loss_fn(pred, target)
        self.total_loss += loss
        dz = self.loss_fn.gradient(pred, target)
        self.model.backward(dz)
        return loss
    def step(self, pred, target):
        raise NotImplementedError

class SGD(OptimizerBase):
    def step(self, pred, target):
        loss = self.compute_gradients(pred, target)
        self.model.update(self.lr)
        return loss
    



# θₜ = θₜ₋₁ - lr * [mₜ / (1 - β₁ᵗ)] / (√[vₜ / (1 - β₂ᵗ)] + ε)
class Adam(OptimizerBase):
    def __init__(self, model, loss_fn, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(model, loss_fn, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = {}
        self.v_w = {}
        self.m_b = {}
        self.v_b = {}
        self.t = 0
    
    def step(self, pred, target):
        loss = self.compute_gradients(pred, target)
        self.t += 1
        
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'dw') and layer.dw is not None:
                # Initialization
                if i not in self.m_w:
                    self.m_w[i] = np.zeros_like(layer.dw)
                    self.v_w[i] = np.zeros_like(layer.dw)
                    self.m_b[i] = np.zeros_like(layer.db)
                    self.v_b[i] = np.zeros_like(layer.db)
                
                # Update Weights
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * layer.dw
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (layer.dw ** 2)
                m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
                v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
                layer.Weights -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                
                # Update Bias
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.db
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.db ** 2)
                m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
                layer.bias -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
        
        return loss