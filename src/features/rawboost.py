#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import copy


def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y

def normWav(x,always):
    if always:
        x = x/np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
            x = x/np.amax(abs(x))
    return x



def genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF,maxF,0);
        bw = randRange(minBW,maxBW,0);
        c = randRange(minCoeff,maxCoeff,1);
          
        if c/2 == int(c/2):
            c = c + 1
        f1 = fc - bw/2
        f2 = fc + bw/2
        if f1 <= 0:
            f1 = 1/1000
        if f2 >= fs/2:
            f2 =  fs/2-1/1000
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs),b)

    G = randRange(minG,maxG,0); 
    _, h = signal.freqz(b, 1, fs=fs)    
    b = pow(10, G/20)*b/np.amax(abs(h))   
    return b


def filterFIR(x,b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N/2):int(y.shape[0]-N/2)]
    return y

# Linear and non-linear convolutive noise
def LnL_convolutive_noise(x,N_f,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,minBiasLinNonLin,maxBiasLinNonLin,fs):
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG-minBiasLinNonLin;
            maxG = maxG-maxBiasLinNonLin;
        b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
        y = y + filterFIR(np.power(x, (i+1)),  b)     
    y = y - np.mean(y)
    y = normWav(y,0)
    return y


# Impulsive signal dependent noise
def ISD_additive_noise(x, P, g_sd):
    beta = randRange(0, P, 0)
    
    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len*(beta/100))
    p = np.random.permutation(x_len)[:n]
    f_r= np.multiply(((2*np.random.rand(p.shape[0]))-1),((2*np.random.rand(p.shape[0]))-1))
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y,0)
    return y


# Stationary signal independent noise

def SSI_additive_noise(x,SNRmin,SNRmax,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise,1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = noise / np.linalg.norm(noise,2) * np.linalg.norm(x,2) / 10.0**(0.05 * SNR)
    x = x + noise
    return x


# ═════════════════════════════════════════════════════════════════════════════
# Project wrapper (added by us — everything above is verbatim from
# github.com/TakHemlata/RawBoost-antispoofing/RawBoost.py)
#
# The upstream wrapper (data_utils_rawboost.py) takes an argparse Namespace.
# We take explicit keyword defaults instead, so preprocess.py stays importable
# without an argparse object. Defaults are the upstream main.py values.
#
# ALGO NUMBERING — upstream definition, do not renumber:
#   1 = LnL convolutive noise
#   2 = ISD impulsive noise
#   3 = SSI coloured additive noise
#   4 = 1 + 2 + 3 in series
#   5 = 1 + 2 in series   <- convolutive + impulsive (LA condition, our choice)
#   6 = 1 + 3 in series
# ═════════════════════════════════════════════════════════════════════════════

# Upstream main.py argparse defaults.
RAWBOOST_DEFAULTS = dict(
    # LnL_convolutive_noise
    N_f=5,
    nBands=5,
    minF=20,
    maxF=8000,
    minBW=100,
    maxBW=1000,
    minCoeff=10,
    maxCoeff=100,
    minG=0,
    maxG=0,
    minBiasLinNonLin=5,
    maxBiasLinNonLin=20,
    # ISD_additive_noise
    P=10,
    g_sd=2,
    # SSI_additive_noise
    SNRmin=10,
    SNRmax=40,
)


def process_Rawboost_feature(feature, sr, algo=5, **overrides):
    """
    Apply RawBoost augmentation to a 1-D float waveform.

    Parameters
    ----------
    feature : np.ndarray, shape [T]
        Raw waveform. Must be 1-D (not [1, T]).
    sr : int
        Sample rate (16000 for this project).
    algo : int
        Which noise process(es) to apply. See ALGO NUMBERING above.
        5 = convolutive + impulsive, the LA-condition setting we use.
        3 = coloured additive, better for the DF/codec condition (Phase 7).
        0 or anything else = passthrough (no augmentation).
    **overrides
        Any key in RAWBOOST_DEFAULTS, to override that hyperparameter.

    Returns
    -------
    np.ndarray, shape [T] — augmented waveform. NOT normalized; the caller
    (preprocess.py) re-normalizes afterwards.
    """
    p = {**RAWBOOST_DEFAULTS, **overrides}
    unknown = set(overrides) - set(RAWBOOST_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown RawBoost hyperparameter(s): {sorted(unknown)}")

    def _conv(x):
        return LnL_convolutive_noise(
            x, p["N_f"], p["nBands"], p["minF"], p["maxF"], p["minBW"], p["maxBW"],
            p["minCoeff"], p["maxCoeff"], p["minG"], p["maxG"],
            p["minBiasLinNonLin"], p["maxBiasLinNonLin"], sr,
        )

    def _impulsive(x):
        return ISD_additive_noise(x, p["P"], p["g_sd"])

    def _coloured(x):
        return SSI_additive_noise(
            x, p["SNRmin"], p["SNRmax"], p["nBands"], p["minF"], p["maxF"],
            p["minBW"], p["maxBW"], p["minCoeff"], p["maxCoeff"],
            p["minG"], p["maxG"], sr,
        )

    chains = {
        1: (_conv,),
        2: (_impulsive,),
        3: (_coloured,),
        4: (_conv, _impulsive, _coloured),
        5: (_conv, _impulsive),
        6: (_conv, _coloured),
    }

    for step in chains.get(algo, ()):
        feature = step(feature)

    return feature


