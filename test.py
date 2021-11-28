#!/usr/bin/env python3

from scipy.io import wavfile
import math
import numpy as np
import numba as nb    
from timeit import default_timer as timer
import argparse

@nb.jit(nopython=True)
def single_goertzel2(samples, sample_rate, freq):
    window_size = len(samples)
    f_step = sample_rate / float(window_size)
    f_step_normalized = 1.0 / window_size

    k = int(math.floor(freq / f_step))
    
    # For all the bins, calculate the DFT term
    n_range = range(0, window_size)

    # Bin frequency and coefficients for the computation
    f = k * f_step_normalized
    w_real = 2.0 * math.cos(2.0 * math.pi * f)
    w_imag = math.sin(2.0 * math.pi * f)

    # Doing the calculation on the whole sample
    d1, d2 = 0.0, 0.0
    for n in n_range:
        y  = samples[n] + w_real * d1 - d2
        d2, d1 = d1, y

    power = d2**2 + d1**2 - w_real * d1 * d2
    
    return (f * sample_rate, power)


def test_freq(sample_rate, data, times, freq):
    WINDOW_SIZE = 4000

    for time in times:
        i = int(time * 8000 / WINDOW_SIZE)
        
        start = timer()
        actual_freq, power = single_goertzel2(data[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE], sample_rate, freq)
        end = timer()
        
        print("time: {}, freq: {}, actual_freq: {}, power: {}".format(time, freq, actual_freq, power))
        print("calc time: {}".format(end - start))


def main():
	parser = argparse.ArgumentParser(description='test frequency detector')
	parser.add_argument('file_name')
	res = parser.parse_args()

	sample_rate, data = wavfile.read(res.file_name)
	print("reading {}, sample_rate: {}, samples: {}".format(res.file_name, sample_rate, len(data)))

	test_times = [0, 4.5, 7, 10, 10.5, 13]
	test_freq(sample_rate, data, test_times, 410)


if __name__ == '__main__':
	main()