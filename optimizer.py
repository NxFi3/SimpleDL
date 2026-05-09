class SGD:
    def __init__(self, model, loss_fn, learning_rate):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = learning_rate
        self.total_loss = 0
    
    def step(self, pred, target):
        loss = self.loss_fn(pred, target)
        self.total_loss += loss
        
        dz = self.loss_fn.gradient(pred, target)
        self.model.backward(dz)
        self.model.update(self.lr)
        
        return loss
    

