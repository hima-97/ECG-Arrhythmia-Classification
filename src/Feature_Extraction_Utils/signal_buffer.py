from array import array
import statistics as stats




# This function is used to safely normalize a value x with respect to its mean x_mean.
# Normalization is a common operation in data preprocessing that scales numeric values to a standard range 
# to ensure that different features contribute equally to a model.
def safe_normalizer(x, x_mean):
    # Prevents division by zero and handles edge cases where x equals its mean.
    if x == x_mean and x != 0:
        return 1
    elif x_mean == 0:
        return x
    else:
        return x / x_mean




# This class is used to manage a buffer of recent signal samples:
# It basically represents a buffer with the last x samples of a signal.
class SignalBuffer:
    
    # Constructor method for the SignalBuffer class:
    def __init__(self, size, initialValue=None):
        self.size = size
        self.empty = False
        if initialValue == None:
            initialValue = 0
            self.empty = True
        self.array = array('f', [initialValue for i in range(self.size * 2)])
        self.index = self.size
        self.sum = 0
        
    # Add a new sample to the buffer:    
    def push(self, x):
        self.sum -= self.array[self.index]
        self.array[self.index] = x
        self.array[self.index - self.size] =  x
        self.sum += x
        self.index += 1
        if self.index >= len(self.array):
            self.index = self.size  
            self.empty = False
            
    # Retrieve the current contents of the buffer:      
    def getBuffer(self):
        return self.array[self.index - self.size : self.index]
    
    # Calculate the mean of the buffer's contents:
    def mean(self):
        if self.empty:
            if self.index == self.size:
                return 0
            return self.sum / (self.index - self.size)
        return self.sum / self.size

    # Calculate the standard deviation of the buffer's contents:
    def std(self):
        if self.empty:
            if self.index <= self.size + 1:
                return 0
            return stats.stdev(self.array[0:self.index - self.size])
        return stats.stdev(self.array[0:self.size])
    
    # Calculate the mean of the most recent 'samples' number of entries in the buffer:
    def partialMean(self, samples):
        if samples <= self.size:
            if self.empty and (samples > self.index - self.size):
                if self.index == self.size:
                    return 0
                samples = self.index - self.size
            partialBuffer = self.getBuffer()[self.size - samples:self.size]
            return stats.mean(partialBuffer)
        else:
            # If the number of samples requested exceeds the size of the buffer,
            # return the mean of the entire buffer.
            return self.mean()
