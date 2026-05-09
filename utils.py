import numpy as np

class OneHotEncoding:
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
    
    def encode(self, y): 
        """number to oneHot
        
        Args:
            y: y = len(num_classes)
            
        Returns:
            numpy array: OneHot Vector
        """
        if isinstance(y, (int, np.integer)):
            one_hot = np.zeros(self.num_classes)
            one_hot[y] = 1
            return one_hot

        return np.array([self.encode(item) for item in y])
    
    
    def decode(self, one_hot):
        if len(one_hot.shape) == 1:
            return np.argmax(one_hot)
        return np.argmax(one_hot, axis=1)
    
    
    def encode_batch(self, y_batch):
       
        return np.array([self.encode(y) for y in y_batch])