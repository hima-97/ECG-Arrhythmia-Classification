import statistics as stats
from array import array
import math


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
        if self.empty and self.index == self.size:
            return 0
        return self.sum / (self.size if not self.empty else (self.index - self.size))

    # Calculate the standard deviation of the buffer's contents:
    def std(self):
        if self.empty and self.index <= self.size + 1:
            return 0
        return stats.stdev(self.array[self.index - self.size:self.index] if self.empty else self.array[:self.size])
    
    # Calculate the mean of the most recent 'samples' number of entries in the buffer:
    def partialMean(self, samples):
        # Calculate the mean of the most recent 'samples' number of entries in the buffer
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




# Function to calculate the derivative of a signal:
def derivative(X, dt):
    dXdt = array('f', [0 for x in range(len(X))])
    for n in range(1, len(X)):
        dXdt[n] = (X[n] - X[n - 1]) / dt
    dXdt[0] = dXdt[1]
    return dXdt


# Function to calculate the zero-crossing points of a signal:
def zeroCrossPoints(X):
    zeroCross = array('B', [False for _ in range(len(X))])
    lastSign = X[0] >= 0
    for n in range(1, len(X)):
        currentSign = X[n] >= 0
        if currentSign != lastSign:
            zeroCross[n] = True
        lastSign = currentSign
    return zeroCross



# Class for QRS detection based on the Pan&Tompkins algorithm:
# The ExtractQRS class is used to extract features related to the QRS complex of each heartbeat.
class ExtractQRS():
    
    # This function
    def __init__(self):
        self.fs = 256  # Adjusted sampling rate (from 150 Hz to 256 Hz)
        # Initialize buffers with adjusted sizes for the new sampling rate:
        self.signalBuffer = SignalBuffer(int(96 * (256 / 150)))
        self.pPeakBuffer = SignalBuffer(int(32 * (256 / 150)))
        self.rPeakBuffer = SignalBuffer(int(32 * (256 / 150)))
        self.qPeakBuffer = SignalBuffer(int(32 * (256 / 150)))
        self.sPeakBuffer = SignalBuffer(int(32 * (256 / 150)))
        self.prWidthBuffer = SignalBuffer(int(32 * (256 / 150)))
        self.qsWidthBuffer = SignalBuffer(int(32 * (256 / 150)))
        self.qrsWidthBuffer = SignalBuffer(int(32 * (256 / 150)))
        self.qrsWidth2Buffer = SignalBuffer(int(32 * (256 / 150)))
        self.qrsWidth4Buffer = SignalBuffer(int(32 * (256 / 150)))
        self.qrsSlopeBuffer = SignalBuffer(int(32 * (256 / 150)))
        self.pqDiffBuffer = SignalBuffer(int(32 * (256 / 150)))
        self.rqDiffBuffer = SignalBuffer(int(32 * (256 / 150)))
        self.rsDiffBuffer = SignalBuffer(int(32 * (256 / 150)))
    
    
    # The findQRSInSignalBuffer method applies the Pan & Tompkins QRS detection algorithm to the buffered signal. 
    # It calculates various features related to the QRS complex, such as the width of the QRS complex, 
    # the amplitude of the Q, R, and S peaks, and the time differences between these peaks. 
    # These features are then returned as a dictionary.
    def findQRSInSignalBuffer(self):
        signal_mean = self.signalBuffer.mean()
        signal = array('f', [s - signal_mean for s in self.signalBuffer.getBuffer()])
        L = len(signal)
        startIndex = L - int(55 * (256/150))  # Adjusted for 256 Hz
        endIndex = L - int(25 * (256/150))  # Adjusted for 256 Hz
        signalMax = max(signal[startIndex:endIndex])
        sMaxIndex = signal[startIndex:endIndex].index(signalMax) + startIndex
        signalMin = min(signal[startIndex:endIndex])
        sMinIndex = signal[startIndex:endIndex].index(signalMin) + startIndex
        signalPeak = signalMax
        signalPeakIndex = sMaxIndex
        if math.fabs(signalMin) > signalPeak:
            signalPeak = signalMin
            signalPeakIndex = sMinIndex

        maxQRSWidth = int(30 * (256/150))  # Adjusted for 256 Hz
        startIndex = int(signalPeakIndex - maxQRSWidth / 2)
        if startIndex < 0:
            startIndex = 0 
        endIndex = int(signalPeakIndex + maxQRSWidth / 2)
        if endIndex >=  L:
            endIndex = L
        dt = 1 / self.fs
        signalDerivative = derivative(signal, dt)
        derivativeZeroCrosses = zeroCrossPoints(signalDerivative)
        zeroCrosses = 0
        qrsStartIndex = startIndex
        qrsWaveform = array('f', [0 for i in range(maxQRSWidth + 1)])
        k = 15
        rPeak = 0
        rIndex = signalPeakIndex
        qPeak = 0
        qIndex = 0
        sIndex = 0
        sPeak = 0
        if signalPeak >= 0:
            rPeak = signalPeak
        lastSample = signalPeak
        halfQrsStartIndex = 0
        quarterQrsStartIndex = 0
        for n in range(signalPeakIndex, startIndex, -1):    
            qrsWaveform[k] = signal[n]
            k -= 1  
            if math.fabs(signal[n]) <= math.fabs(signalPeak) / 2 and halfQrsStartIndex == 0:
                halfQrsStartIndex = k
            if math.fabs(signal[n]) <= math.fabs(signalPeak) / 4 and quarterQrsStartIndex == 0:
                quarterQrsStartIndex = k
            if derivativeZeroCrosses[n] and n < signalPeakIndex - 1:
                zeroCrosses += 1
                if zeroCrosses == 1 and rPeak > 0:
                    # First peak before R peak
                    if signal[n-1] < 0:
                        # If negative then it's a Q peak
                        qPeak = signal[n-1]
                        qIndex = n - 1                    
                elif zeroCrosses == 1 and rPeak == 0:
                    # First peak before a main negative peak (Q or S)
                    if signal[n-1] > 0:
                        # It's a R peak
                        rPeak = signal[n - 1]
                        rIndex = n -1
                        # Main peak is a S peak
                        sPeak = signalPeak
                        sIndex = signalPeakIndex
                    else:
                        # Main peak is a Q peak
                        qPeak = signalPeak
                        qIndex = signalPeakIndex
                if zeroCrosses == 2:
                    if qPeak == 0 and signal[n-1] < 0:
                        qPeak = signal[n-1]
                        qIndex = n - 1
                    else:
                        # Found a second positive peak before R or Q Peak
                        qrsStartIndex = n -1
                        break
            
            lastSample = signal[n]
            
        if qIndex == 0:
            qIndex = qrsStartIndex
        qrsEndIndex = endIndex
        k = 16
        zeroCrosses = 0
        lastSample = signalPeak
        qrsEndReady = False
        halfQrsEndIndex = 30
        quarterQrsEndIndex = 30
        for n in range(signalPeakIndex + 1, endIndex):  
            qrsWaveform[k] = signal[n]
            k += 1
            if math.fabs(signal[n]) <= math.fabs(signalPeak) / 2 and halfQrsEndIndex == 30:
                halfQrsEndIndex = k
            if math.fabs(signal[n]) <= math.fabs(signalPeak) / 2 and quarterQrsEndIndex == 30:
                quarterQrsEndIndex = k
            if qrsEndReady == True and signal[n] * lastSample <= 0:
                # Found a signal sign change (zero cross) while ready to finish
                qrsEndIndex = n
                break
            if derivativeZeroCrosses[n] and n > signalPeakIndex + 2:
                zeroCrosses += 1
                if zeroCrosses == 1 and rPeak == signalPeak:
                    # First peak after R peak
                    if signal[n-1] < 0:
                        # If negative then it's a S peak
                        sPeak = signal[n-1]
                        sIndex = n -1 
                # Otherwise assume QRS is over
                qrsEndReady = True
                if zeroCrosses == 2:
                    # Found a second peak after R or S so assume it's the end
                    qrsEndIndex = n -1
                    break
            
            lastSample = signal[n]
        if sIndex == 0:
            sIndex = qrsEndIndex
        # P wave finding
        pStart = qrsStartIndex - int(35 * (256/150))  # Adjusted for 256 Hz
        if pStart < int(10 * (256/150)):  # Adjusted for 256 Hz
            pStart = int(10 * (256/150))  # Adjusted for 256 Hz
        pEnd = qrsStartIndex - int(10 * (256/150))  # Adjusted for 256 Hz
        baseStart = pStart - int(10 * (256/150))  # Adjusted for 256 Hz
        if pEnd < int(20 * (256/150)):  # Adjusted for 256 Hz
            pEnd = int(20 * (256/150))  # Adjusted for 256 Hz
        std_noise = stats.stdev(signal[baseStart:pStart])
        mean_noise = stats.mean(signal[baseStart:pStart])
        pPeak = max(signal[pStart:pEnd])
        pIndex = signal[pStart:pEnd].index(pPeak) + pStart
        if pPeak < mean_noise + 3 * std_noise or pIndex == pStart or pIndex == pEnd:
            pPeak = 0
            pIndex = qrsStartIndex
        qsWidth = (sIndex - qIndex) * dt
        prWidth = (qrsStartIndex - pIndex) * dt
        maxSlope = abs(max(signalDerivative[qrsStartIndex:qrsEndIndex]))
        qrsWidth = (qrsEndIndex - qrsStartIndex) * dt
        halfQrsWidth = (halfQrsEndIndex - halfQrsStartIndex) * dt
        quarterQrsWidth = (quarterQrsEndIndex - quarterQrsStartIndex) * dt
        if False: #Debug
            import matplotlib.pyplot as plt
            plt.plot(signal, '-', pIndex, pPeak, 'r*', qIndex, qPeak, 'ro', rIndex, rPeak, 'rx', sIndex,
                     sPeak, 'r+', qrsStartIndex, signal[qrsStartIndex], 'g.', qrsEndIndex, signal[qrsEndIndex], 'g.')
            plt.show()
        
        pqDiff = pPeak - qPeak
        rqDiff = rPeak - qPeak
        rsDiff = rPeak - sPeak
        self.pPeakBuffer.push(pPeak)
        self.rPeakBuffer.push(rPeak)
        self.qPeakBuffer.push(qPeak)
        self.sPeakBuffer.push(sPeak)
        self.prWidthBuffer.push(prWidth)
        self.qsWidthBuffer.push(qsWidth)
        self.qrsWidthBuffer.push(qrsWidth)
        self.qrsWidth2Buffer.push(halfQrsWidth)
        self.qrsWidth4Buffer.push(quarterQrsWidth)
        self.qrsSlopeBuffer.push(maxSlope)
        self.pqDiffBuffer.push(pqDiff)
        self.rqDiffBuffer.push(rqDiff)
        self.rsDiffBuffer.push(rsDiff)
        return {
            'QRSw': qrsWidth, 
            'QRSw2': halfQrsWidth,
            'QRSw4': quarterQrsWidth,
            'QSw': qsWidth,
            'PRw': prWidth,
            'Ppeak': pPeak,
            'Rpeak': rPeak,
            'Qpeak': qPeak,
            'Speak': sPeak,
            'PQa': pqDiff,
            'RQa': rqDiff,
            'RSa': rsDiff,
            'Ppeak_norm': safe_normalizer(pPeak,self.pPeakBuffer.mean()),
            'Rpeak_norm': safe_normalizer(rPeak,self.rPeakBuffer.mean()),
            'Qpeak_norm': safe_normalizer(qPeak,self.qPeakBuffer.mean()),
            'Speak_norm': safe_normalizer(sPeak,self.sPeakBuffer.mean()),
            'PQa_norm': safe_normalizer(pqDiff, self.pqDiffBuffer.mean()),
            'RQa_norm': safe_normalizer(rqDiff, self.rqDiffBuffer.mean()),
            'RSa_norm': safe_normalizer(rsDiff, self.rsDiffBuffer.mean()),
            'PRa_norm': safe_normalizer(prWidth, self.prWidthBuffer.mean()),
            'QSa_norm': safe_normalizer(qsWidth, self.qsWidthBuffer.mean()),
            'QRSw_norm': safe_normalizer(qrsWidth, self.qrsWidthBuffer.mean()),
            'QRSw2_norm': safe_normalizer(halfQrsWidth, self.qrsWidth2Buffer.mean()),
            'QRSw4_norm': safe_normalizer(quarterQrsWidth, self.qrsWidth4Buffer.mean()),
            'QRSs': maxSlope, 
            'QRSs_norm': safe_normalizer(maxSlope, self.qrsSlopeBuffer.mean())
        }

    # This `__call__` method allows an instance of the class to be called like a function. 
    # This method takes a beat time and a signal as input, pushes the samples around the beat time into a signal buffer, 
    # and then calls the findQRSInSignalBuffer method to extract features from the QRS complex in the buffered signal. 
    def __call__(self, beatTime, signal):
        beatSample = int(beatTime * 256)
        # Adjust the window size around the R peak for 256 Hz
        pre_samples = int(128 * (256 / 150))  # Adjusted for 256 Hz
        post_samples = int(40 * (256 / 150))  # Adjusted for 256 Hz
        for n in range(beatSample - pre_samples, beatSample + post_samples):
            rawSample = signal[n]
            self.signalBuffer.push(rawSample)
        return self.findQRSInSignalBuffer()

                        