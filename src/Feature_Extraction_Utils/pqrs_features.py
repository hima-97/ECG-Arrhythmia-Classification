from array import array
import math
from .signal_buffer import SignalBuffer
from .signal_buffer import safe_normalizer
import statistics as stats


# This file is dedicated to extacting features from the QRS complex of each heartbeat.
# the following temporal characteristics of the QRS complex are calculated: 
# the total duration of the QRS complex (𝑄𝑅𝑆𝑤), 
# the width of the QRS complex at half of the peak value (𝑄𝑅𝑆𝑤2), 
# the width of the QRS complex at a quarter of the peak value (𝑄𝑅𝑆𝑤4), 
# the distance between the peak of the Q wave, 
# and the peak of the S wave (𝑄𝑆𝑑).


# A detailed description of the extraction procedure is provided below:

# The R spike annotations provided with the MIT-BIH Arrhythmia Database are used as a marker to separate and identify the beats. 
# At each beat location, a segment of 640 ms of signal (164 samples for 256 Hz) is considered, 
# 373 ms before the annotation (95 samples for 256 Hz), and 267 ms (68 samples for 256 Hz) after it. 
# The mean of the signal segment is subtracted from each sample in order to remove the baseline. 
# The absolute maximum value of the signal 100 ms before and after the annotation (𝑄𝑅𝑆𝑚𝑎𝑥) is considered as a reference point. 
# Though such value usually corresponds to the database R spike annotation and the peak of the R wave in typical normal heartbeats, 
# this is not always the case because the QRS may have a complex morphology 
# and fall into one of the broad categories of the standard nomenclature where the R wave is not the one with the highest amplitude [39]. 
# Therefore, only if 𝑄𝑅𝑆𝑚𝑎𝑥 is positive is it immediately marked as the peak of the R wave (𝑅𝑝𝑒𝑎𝑘). 
# Starting from 𝑄𝑅𝑆𝑚𝑎𝑥, ten fiducial points are identified: the two locations where the signal reaches half the value of 𝑄𝑅𝑆𝑚𝑎𝑥 (𝑄𝑅𝑆𝑚𝑎𝑥/2𝑎,𝑏), 
# the two locations where the signal reaches a quarter of the value of 𝑄𝑅𝑆𝑚𝑎𝑥 (𝑄𝑅𝑆𝑚𝑎𝑥/4𝑎,𝑏), the peak value of the R wave (𝑅𝑝𝑒𝑎𝑘), the peak value of the Q wave (𝑄𝑝𝑒𝑎𝑘), 
# the peak value of the S wave (𝑆𝑝𝑒𝑎𝑘), the beginning of the QRS complex (𝑄𝑅𝑆𝑠𝑡𝑎𝑟𝑡), the end of the QRS complex (𝑄𝑅𝑆𝑒𝑛𝑑), and the peak value of the P wave (𝑃𝑝𝑒𝑎𝑘). 

# Many of the mentioned fiducial points are individuated by looking for the inflection points in the signal 
# and identifying the locations where the signal's first derivative changes direction. For this, a two-point numerical differentiation is applied to the signal. 
# The procedure for identifying the fiducial points can be described with the following steps:

# 1) Assume that 𝑄𝑝𝑒𝑎𝑘, 𝑅𝑝𝑒𝑎𝑘, 𝑆𝑝𝑒𝑎𝑘, and 𝑃𝑝𝑒𝑎𝑘 equal zero and that the corresponding waves are not present.

# 2) If 𝑄𝑅𝑆𝑚𝑎𝑥 is positive, then make 𝑅𝑝𝑒𝑎𝑘 equal to 𝑄𝑅𝑆𝑚𝑎𝑥.

# 3) Look backward from 𝑄𝑅𝑆𝑚𝑎𝑥 and evaluate the signal and its inflection points in this way:
# (a) Make 𝑄𝑅𝑆𝑚𝑎𝑥/2𝑎 equal to the first location where the signal goes below half of 𝑄𝑅𝑆𝑚𝑎𝑥.
# (b) Make 𝑄𝑅𝑆𝑚𝑎𝑥/4𝑎 equal to the first location where the signal goes below a quarter of 𝑄𝑅𝑆𝑚𝑎𝑥.
# (c) If the first inflection point is negative and 𝑅𝑝𝑒𝑎𝑘 is not zero, then 𝑄𝑝𝑒𝑎𝑘 equals the value at such point.
# (d) If the first inflection point is positive or zero and 𝑅𝑝𝑒𝑎𝑘 is not zero, then it is marked as 𝑄𝑅𝑆𝑠𝑡𝑎𝑟𝑡, and 𝑄𝑝𝑒𝑎𝑘 is considered zero.
# (e) If the first inflection point is positive and 𝑅𝑝𝑒𝑎𝑘 is zero, then make 𝑅𝑝𝑒𝑎𝑘 equal to the value at such point and make 𝑆𝑝𝑒𝑎𝑘 equal to QRS max.
# (f) If the second inflection point is negative, 𝑄𝑝𝑒𝑎𝑘 is zero, and 𝑄𝑅𝑆𝑚𝑎𝑥 is positive, then make 𝑄𝑝𝑒𝑎𝑘 equal to the value at such point.
# (g) If 𝑄𝑝𝑒𝑎𝑘 is not zero and the signal crosses zero, then the first non-negative point is marked as 𝑄𝑅𝑆𝑠𝑡𝑎𝑟𝑡.
# (h) If the second inflection point is positive or zero and 𝑄𝑅𝑆𝑠𝑡𝑎𝑟𝑡 has not been found yet, then it is marked as 𝑄𝑅𝑆𝑠𝑡𝑎𝑟𝑡.

# 4) Look forward from 𝑄𝑅𝑆𝑚𝑎𝑥 and evaluate the signal and its inflection points in this way:
# (a) Make 𝑄𝑅𝑆𝑚𝑎𝑥/2𝑏 equal to the first location where the signal goes below half of 𝑄𝑅𝑆𝑚𝑎𝑥.
# (b) Make 𝑄𝑅𝑆𝑚𝑎𝑥/4𝑏 equal to the first location where the signal goes below a quarter of 𝑄𝑅𝑆𝑚𝑎𝑥.
# (c) If the first inflection point is negative and 𝑅𝑝𝑒𝑎𝑘 is not zero, then make 𝑆𝑝𝑒𝑎𝑘 equal to the value at such point.
# (d) If 𝑆𝑝𝑒𝑎𝑘 is not zero and the signal cross zero, then the first non-negative point is marked as 𝑄𝑅𝑆𝑒𝑛𝑑.
# (e) If the second inflection point is positive or zero and 𝑄𝑅𝑆𝑒𝑛𝑑 has not been found yet, then it is marked as 𝑄𝑅𝑆𝑒𝑛𝑑.

# 5) Find the maximum value of the signal in the segment that goes between 233 ms (60 samples for 256 Hz) and 67 ms (17 samples for 256 Hz) before 𝑄𝑅𝑆𝑠𝑡𝑎𝑟𝑡. 
# If such value is greater than three times, the standard deviation of the signal during the 67 ms preceding the segment in consideration, 
# and its position corresponds to an inflection point in the signal, then make 𝑃𝑝𝑒𝑎𝑘 equal to such a value.


# Function to calculate the derivative of a signal:
def derivative(X, dt):
    dXdt = array('f', [0 for x in range(len(X))])
    for n in range(1, len(X)):
        dXdt[n] = (X[n] - X[n - 1]) / dt # / 20 (when ploting)
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
        self.signalBuffer = SignalBuffer(164) # Adjusted buffer size for 256 Hz (164 samples = 640 ms)
        # The class uses several SignalBuffer objects to store different features of the QRS complex:
        # Each buffer stores a certain number of the most recent values of a particular feature.
        # The SignalBuffer size of 32 for most of the buffers likely represents the number of heartbeats used to calculate certain features.
        self.pPeakBuffer = SignalBuffer(32)
        self.rPeakBuffer = SignalBuffer(32)
        self.qPeakBuffer = SignalBuffer(32)
        self.sPeakBuffer = SignalBuffer(32)
        self.prWidthBuffer = SignalBuffer(32)
        self.qsWidthBuffer = SignalBuffer(32)
        self.qrsWidthBuffer = SignalBuffer(32)
        self.qrsWidth2Buffer = SignalBuffer(32)
        self.qrsWidth4Buffer = SignalBuffer(32)
        self.qrsSlopeBuffer = SignalBuffer(32)
        self.pqDiffBuffer = SignalBuffer(32)
        self.rqDiffBuffer = SignalBuffer(32)
        self.rsDiffBuffer = SignalBuffer(32)
    
    
    # The findQRSInSignalBuffer method primarily applies the Pan & Tompkins QRS detection algorithm to the buffered signal. 
    # It calculates various features related to the QRS complex, such as the width of the QRS complex, 
    # the amplitude of the Q, R, and S peaks, and the time differences between these peaks. 
    # These features are then returned as a dictionary.
    def findQRSInSignalBuffer(self):
        
        # Calculate the mean of the buffered signal:
        signal_mean = self.signalBuffer.mean()
        
        # Normalizing the signal by subtracting the mean, making the signal centered around zero:
        signal = array('f', [s - signal_mean for s in self.signalBuffer.getBuffer()])
        
        # This is where you're looking for the maximum and minimum values in a specific segment of the signal to identify QRSmax:
        # This window is chosen based on the expected location of QRS complexes.
        # Note: There is a window of 51 samples (i.e. 200 ms) between the start and end indices.
        # This means that the absolute maximum value of the signal 100 ms before and 100 ms after the annotation (𝑄𝑅𝑆𝑚𝑎𝑥) is considered as a reference point.
        L = len(signal)
        startIndex = L - int(55 * (256 / 150)) # Adjusted index for 256 Hz                
        endIndex = L - int(25 * (256 / 150)) # Adjusted index for 256 Hz    
        signalMax = max(signal[startIndex:endIndex])
        sMaxIndex = signal[startIndex:endIndex].index(signalMax) + startIndex
        signalMin = min(signal[startIndex:endIndex])
        sMinIndex = signal[startIndex:endIndex].index(signalMin) + startIndex
        
        # The QRSmax is then determined as the absolute highest peak within the window, whether it is a maximum or a minimum peak:
        # Here `signalPeak` represents QRSmax, and `signalPeakIndex` represents the index of QRSmax.
        signalPeak = signalMax
        signalPeakIndex = sMaxIndex
        if math.fabs(signalMin) > signalPeak:
            signalPeak = signalMin
            signalPeakIndex = sMinIndex


        # `maxQRSWidth` is set to 30, which is used to define a window around the signal peak (maximum or minimum).
        # This window is assumed to contain the QRS complex.
        # The window's start and end points are adjusted to ensure they don't exceed the boundaries of the signal.
        # The derivative of the signal is calculated, which is used for finding inflection points.
        # `derivativeZeroCrosses` stores the locations where the derivative of the signal crosses zero.
        # A waveform (`qrsWaveform`) is initialized to store a part of the signal around the QRS complex.

        maxQRSWidth = 51 # Adjusted window size for 256 Hz (corresponds to 200 ms)
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
        
        # Initializing `qrsWaveform`:
        # This is an array that stores a portion of the ECG signal, specifically the segment encompassing the QRS complex and some surrounding signal.
        # Array of floats ('f') with 52 elements, each initialized to 0
        # The length of 52 is chosen because it's one more than the maxQRSWidth of 51, which represents the window around the QRS peak in which the algorithm is interested.
        qrsWaveform = array('f', [0 for i in range(52)])
        # Index for the middle of the array (note: arrays are 0-indexed):
        k = 25 # Adjusted for 256 Hz (25 is half of 51, the maxQRSWidth)
               # k is rounded down to 25 because of bacward analysis from QRSmax

        
        # 1) Assuming Qpeak, Rpeak, Speak, and Ppeak Equal Zero (Initial Assumption):
        rPeak = 0
        rIndex = signalPeakIndex
        qPeak = 0
        qIndex = 0
        sIndex = 0
        sPeak = 0
        
        # 2) If QRSmax is Positive, Make Rpeak Equal to QRSmax:
        if signalPeak >= 0:
            rPeak = signalPeak
        lastSample = signalPeak
        halfQrsStartIndex = 0
        quarterQrsStartIndex = 0
        
        # 3) Looking Backward from QRSmax and evaluating the signal and its inflection points:
        for n in range(signalPeakIndex, startIndex, -1):    
            qrsWaveform[k] = signal[n]
            k -= 1
            # 3a) Make QRSmax/2a Equal to the First Location Where the Signal Goes Below Half of QRSmax:
            if math.fabs(signal[n]) <= math.fabs(signalPeak) / 2 and halfQrsStartIndex == 0:
                halfQrsStartIndex = k
            # 3b) Make QRSmax/4a Equal to the First Location Where the Signal Goes Below a Quarter of QRSmax:
            if math.fabs(signal[n]) <= math.fabs(signalPeak) / 4 and quarterQrsStartIndex == 0:
                quarterQrsStartIndex = k
            # Handling different cases based on zero crossings and the state of Rpeak:
            if derivativeZeroCrosses[n] and n < signalPeakIndex - 1:
                zeroCrosses += 1
                # 3c) If the first inflection point is negative and Rpeak is not zero, then Qpeak equals the value at such point.
                if zeroCrosses == 1 and rPeak > 0:
                    # First peak before R peak
                    if signal[n-1] < 0:
                        # If negative then it's a Q peak
                        qPeak = signal[n-1]
                        qIndex = n - 1
                # 3e) If the first inflection point is positive and Rpeak is zero, then make Rpeak equal to the value at such point and make Speak equal to QRSmax.                    
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
                # 3f) If the second inflection point is negative, Qpeak is zero, and QRSmax is positive, then make Qpeak equal to the value at such point.
                # 3g) Handling the second negative peak when Qpeak is zero, marking QRSstart.
                # 3h) If the second inflection point is positive or zero, mark it as QRSstart.
                if zeroCrosses == 2:
                    if qPeak == 0 and signal[n-1] < 0:
                        qPeak = signal[n-1]
                        qIndex = n - 1
                    else:
                        qrsStartIndex = n -1
                        break
            
            lastSample = signal[n]
            
        # Check if Q peak index (qIndex) is not set (i.e., remains 0). If it is, then set it to the QRS start index:
        if qIndex == 0:
            qIndex = qrsStartIndex

        # Setting the end index of the QRS complex analysis window:
        qrsEndIndex = endIndex
        # Resetting variables for forward analysis from QRSmax:
        k = 26  # Adjusted for 256 Hz (26 is half of 51, the maxQRSWidth)
                # k is rounded up this time to 26 because of forward analysis from QRSmax
        zeroCrosses = 0  # Counter for the number of zero crossings found in the forward search
        lastSample = signalPeak  # Last sample analyzed, initialized to signalPeak
        qrsEndReady = False  # Flag to indicate readiness to mark the end of the QRS complex
        halfQrsEndIndex = 51 # Adjusted index for 256 Hz
        quarterQrsEndIndex = 51 # Adjusted index for 256 Hz
                
        # 4) Looking Forward from QRSmax and evaluating the signal and its inflection points:
        for n in range(signalPeakIndex + 1, endIndex):  
            qrsWaveform[k] = signal[n]
            k += 1
            # 4a) Make QRSmax/2b Equal to the First Location Where the Signal Goes Below Half of QRSmax:
            if math.fabs(signal[n]) <= math.fabs(signalPeak) / 2 and halfQrsEndIndex == 51:
                halfQrsEndIndex = k
            # 4b) Make QRSmax/4b Equal to the First Location Where the Signal Goes Below a Quarter of QRSmax:
            if math.fabs(signal[n]) <= math.fabs(signalPeak) / 2 and quarterQrsEndIndex == 51:
                quarterQrsEndIndex = k
            # 4c) Identifying the S peak if the first inflection point is negative and Rpeak is not zero:
            if qrsEndReady == True and signal[n] * lastSample <= 0:
                # Found a signal sign change (zero cross) while ready to finish
                qrsEndIndex = n
                break
            # 4d) & 4e) Determining QRSend based on signal crossing zero and second inflection point:
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
        
        # Handling case where S peak is not identified in the forward loop:
        if sIndex == 0:
            sIndex = qrsEndIndex
            
        # 5) Finding Ppeak: Determine maximum value of the signal in the segment before QRSstart:
        pStart = qrsStartIndex - 60 # Adjusted index for 256 Hz (60 samples = 233 ms)
        if pStart < 17: # Adjusted index for 256 Hz (17 samples = 67 ms)
            pStart = 17
        pEnd = qrsStartIndex - 17
        baseStart = pStart - 17
        if pEnd < 34:
            pEnd = 34
        std_noise = stats.stdev(signal[baseStart:pStart])
        mean_noise = stats.mean(signal[baseStart:pStart])
        pPeak = max(signal[pStart:pEnd])
        pIndex = signal[pStart:pEnd].index(pPeak) + pStart
        
        # Validate Ppeak based on noise thresholding:
        if pPeak < mean_noise + 3 * std_noise or pIndex == pStart or pIndex == pEnd:
            pPeak = 0
            pIndex = qrsStartIndex
            
        # Calculate various widths and differences between peaks:
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
        
        
        # Calculate differences between various peaks:
        pqDiff = pPeak - qPeak
        rqDiff = rPeak - qPeak
        rsDiff = rPeak - sPeak
        
        # Update buffers with the newly calculated values:
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
        
        # Return a dictionary with the calculated features related to the QRS complex:
        # This dictionary returns indices that are relative to the segment of the signal that was analyzed.
        return {
            'QRSstart': qrsStartIndex,
            'QRSend': qrsEndIndex,
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
            # Normalized values of the features:
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
        
        # beatTime is multiplied by 256 to convert the time of the beat (in seconds) to an index in the sampled signal:
        beatSample = int(beatTime * 256)
        
        # At each beat location, a segment of 640 ms of signal (164 samples for 256 Hz) is considered, 
        # 373 ms before the annotation (95 samples for 256 Hz), and 267 ms (68 samples for 256 Hz) after it. 
        # Adjust the window size around the R peak for 256 Hz.
        pre_samples = int(128 * (256 / 150))  # 95 samples for 256 Hz
        post_samples = int(40 * (256 / 150))  # 68 samples for 256 Hz
        
        for n in range(beatSample - pre_samples, beatSample + post_samples):
            rawSample = signal[n]
            self.signalBuffer.push(rawSample)
            
            
        return self.findQRSInSignalBuffer()