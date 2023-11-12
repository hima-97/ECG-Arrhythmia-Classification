import statistics as stats
from .signal_buffer import SignalBuffer
from .signal_buffer import safe_normalizer


# This file is used to extract R-R interval related features.


class RRFeatures():

    def __init__(self):
        self.rrBuffer = SignalBuffer(int(32 * (256 / 150)))

    def __call__(self, beats, beat_index):
        labeledBeatTime = beats[beat_index]['time']
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
        if beat_index < (int(32 * (256 / 150))):
            startbeat_index = 0
        else:
            startbeat_index = beat_index - (int(32 * (256 / 150)))
        rrBuffer = []
        for k in range(startbeat_index, beat_index):
            rrBuffer.append(beats[k + 1]
                            ['time'] - beats[k]['time'])

        previousRR = lastLabeledBeatTime - prevLabeledBeatTime
        currentRR = labeledBeatTime - lastLabeledBeatTime
        nextRR = nextLabeledBeatTime - labeledBeatTime
        self.rrBuffer.push(currentRR)
        averageRR = self.rrBuffer.mean()
        stddevRR = self.rrBuffer.std()
        if stddevRR == 0:
            student = 0
        else:
            student = (currentRR - averageRR) / stddevRR
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
