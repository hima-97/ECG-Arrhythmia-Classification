import statistics as stats
from .signal_buffer import SignalBuffer
from .signal_buffer import safe_normalizer


# This file is used to extract R-R interval related features.

# Normalized heart rate features (6 features): 
# 𝑅𝑅0 divided by the average of the last 32 beats (𝑅𝑅0/𝑎𝑣𝑔𝑅𝑅), 
# 𝑅𝑅−1 divided by the average of the last 32 R–R intervals (𝑅𝑅−1/𝑎𝑣𝑔𝑅𝑅), 
# 𝑅𝑅+1 divided by the average of the last 32 R–R intervals (𝑅𝑅+1/𝑎𝑣𝑔𝑅𝑅), 
# 𝑅𝑅−1 divided by 𝑅𝑅0 (𝑅𝑅−1/𝑅𝑅0), 𝑅𝑅+1 divided by 𝑅𝑅0 (𝑅𝑅+1/𝑅𝑅0), 
# and the t-statistic of 𝑅𝑅0 (𝑡𝑅𝑅0) defined by the difference between 𝑅𝑅0 and 𝑎𝑣𝑔𝑅𝑅 divided by the standard deviation of the last 32 R–R intervals.

# Normalized QRS temporal characteristics and amplitude differences (12 features): 
# the same QRS temporal characteristics and amplitude differences previously specified, except that they are divided by their average value in the last 32 heartbeats.



class RRFeatures():

    # This is the constructor method for the RRFeatures class:
    # It's called when an object of this class is instantiated.
    def __init__(self):
        # This creates a SignalBuffer object with a size of 32 and assigns it to the rrBuffer attribute of the RRFeatures object:
        # This is independent of the sampling rate of the ECG signal.
        # That is because the buffer is used to store the last 32 R-R intervals. 
        # This size is not related to the sampling rate, but rather to the number of heartbeats you want to consider for your calculations.
        self.rrBuffer = SignalBuffer(32)

    # This method is called when the class instance is "called" like a function:
    # See `feature_extraction.py` for an example of how this class is used.
    def __call__(self, beats, beat_index):
        
        labeledBeatTime = beats[beat_index]['time'] # This gets the 'time' value of the beat at the given index.
        
        # These lines calculate the times of the previous, current, and next beats based on the given beat index:
        if beat_index < 2:
            prevLabeledBeatTime = 0
        else:
            prevLabeledBeatTime = beats[beat_index - 2]['time']
        if beat_index == 0:
            lastLabeledBeatTime = 0
        else:
            lastLabeledBeatTime = beats[beat_index - 1]['time']
        if beat_index + 1 == len(beats):
            nextLabeledBeatTime = beats[-1]['time']
        else:
            nextLabeledBeatTime = beats[beat_index + 1]['time']
        if beat_index < 32:
            startbeat_index = 0
        else:
            startbeat_index = beat_index - 32
            
        # This loop calculates the R-R intervals for the last 32 beats (or fewer if there are less) and stores them in rrBuffer:
        rrBuffer = []
        for k in range(startbeat_index, beat_index):
            rrBuffer.append(beats[k + 1]
                            ['time'] - beats[k]['time'])

        # Calculating the previous, current, and next RR intervals:
        previousRR = lastLabeledBeatTime - prevLabeledBeatTime
        currentRR = labeledBeatTime - lastLabeledBeatTime
        nextRR = nextLabeledBeatTime - labeledBeatTime
        
        self.rrBuffer.push(currentRR) # Pushing the current RR interval to the rrBuffer
        averageRR = self.rrBuffer.mean() # Calculating the mean of the R-R intervals in the rrBuffer
        stddevRR = self.rrBuffer.std() # Calculating the standard deviation of the R-R intervals in the rrBuffer
        
        if stddevRR == 0:
            student = 0
        else:
            student = (currentRR - averageRR) / stddevRR
            
        # Return a dictionary containing the calculated features:
        return {
            'RR0': currentRR,
            'RR-1': previousRR,
            'RR+1': nextRR,
            'RR0/avgRR': safe_normalizer(currentRR, averageRR),
            'tRR0': student,
            'RR-1/avgRR': safe_normalizer(previousRR, averageRR),
            'RR-1/RR0': safe_normalizer(previousRR, currentRR),
            'RR+1/avgRR': safe_normalizer(nextRR, averageRR),
            'RR+1/RR0': safe_normalizer(nextRR, currentRR),
        }
