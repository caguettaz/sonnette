#!/usr/bin/env python3

from scipy.io import wavfile
import math
import numpy as np
import numba as nb    
from timeit import default_timer as timer
import argparse

@nb.jit(nopython=True)
def single_goertzel2(samples, sample_rate, freq):

    w_real = 2.0 * math.cos(2.0 * math.pi * freq / sample_rate)
    w_imag = math.sin(2.0 * math.pi * freq / sample_rate)

    # Doing the calculation on the whole sample
    d1, d2 = 0.0, 0.0
    for s in samples:
        y  = s + w_real * d1 - d2
        d2, d1 = d1, y

    power = d2**2 + d1**2 - w_real * d1 * d2
    
    return power


def test_freq(sample_rate, data, times, freq):
    WINDOW_SIZE = 4000

    for time in times:
        i = int(time * 8000 / WINDOW_SIZE)
        
        start = timer()
        power = single_goertzel2(data[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE], sample_rate, freq)
        end = timer()
        
        print("time: {}, freq: {}, power: {}".format(time, freq, power))
        print("calc time: {}".format(end - start))


def test_freqs(sample_rate, data, time, freqs):
    WINDOW_SIZE = 4000
    i = int(time * 8000 / WINDOW_SIZE)

    for freq in freqs:
        power = single_goertzel2(data[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE], sample_rate, freq)

        logp = math.log(power)
        print("time: {}, freq: {}, power: {}".format(time, freq, logp))

def score_data(sample_rate, data):
    pos_freqs = [150, 287, 567, 1281, 1531]
    neg_freqs = [427, 924, 1406, 2000]

    pos_scores = [math.log(single_goertzel2(data, sample_rate, f)) for f in pos_freqs]
    neg_scores = [math.log(single_goertzel2(data, sample_rate, f)) for f in neg_freqs]

    if max(pos_scores) < max(neg_scores) or min(pos_scores) < min(neg_scores):
        return -math.inf

    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    pos_scores = pos_scores - min(neg_scores)
    neg_scores = neg_scores - min(neg_scores)

    neg_scores = neg_scores / max(pos_scores)
    pos_scores = pos_scores / max(pos_scores)

    pos_signature = np.array([1., 0.62, 0.65, 0.7, 0.86])
    neg_signature = np.array([0.4, 0.07, 0.,  0.])

    return -math.sqrt(sum((pos_scores - pos_signature) ** 2)) - \
            math.sqrt(sum((neg_scores - neg_signature) ** 2))


def main():
    parser = argparse.ArgumentParser(description='test frequency detector')
    parser.add_argument('file_name')
    res = parser.parse_args()

    sample_rate, data = wavfile.read(res.file_name)
    print("reading {}, sample_rate: {}, samples: {}".format(
        res.file_name, sample_rate, len(data)))

    test_times = [0, 4.5, 7, 10, 10.5, 13]

    for t10 in range(0, 130):
        t = t10 / 10
        WINDOW_SIZE = 4000
        i = int(t * 8000 / WINDOW_SIZE)

        print(f"**********{t}***********")
        score = score_data(sample_rate, data[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE])

        print(score)

    #test_freq(sample_rate, data, test_times, 3634)

    # freqs = [150, 287, 567, 1281, 1531]

    # interp_freqs = freqs[0:1]
    # for f in freqs[1:]:
    #     interp_freqs.extend([(f + interp_freqs[-1]) / 2, f])

    # interp_freqs.append(2000)

    # for t in test_times:
    #     print(f"**********{t}***********")
    #     test_freqs(sample_rate, data, t, interp_freqs)


if __name__ == '__main__':
    main()