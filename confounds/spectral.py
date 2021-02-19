"""
Mini-module containing confound methods to perform spectral
cleaning of fMRI signals
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import numpy.typing as npt

# TODO: Use Numba

def make_design(t: npt.ArrayLike,
                w: npt.ArrayLike,
                o: Optional[npt.ArrayLike] = None,
                n_fourier_exp: Optional[int] = 1) -> npt.ArrayLike:
    '''
    Construct a sinusoidal design matrix phase shifted by
    frequency dependent map o(w)

    Args:
        t: Sampling times [T x 1]
        w: Frequencies to include (samples/sec) [num freqs x 1]
        o: Frequency dependent map, must match length of w [num freqs x 1]
        n_fourier_exp: Number of fourier expansions

    Returns:
        [len(t) x (2*n_fourier_exp*w.shape[0])] Design matrix
    '''

    stride = 2 * w.shape[0]
    X = np.empty((t.shape[0], stride * n_fourier_exp))

    if o is None:
        o = np.zeros_like(w)

    for i in np.arange(0, n_fourier_exp):
        X[:, i * stride:stride * (i + 1):2] = np.cos(w * (i + 1) * (t - o))
        X[:, (i * stride) + 1:(stride * (i + 1)) + 1:2] = np.sin(w * (i + 1) *
                                                                 (t - o))

    return X


def get_sampling_w(t: npt.ArrayLike,
                   oversampling: Optional[int] = 8,
                   max_freq: Optional[int] = 1) -> npt.ArrayLike:
    '''
    Get sampling frequency of time-series

    Args:
        t: Sampling times
        oversampling: Oversampling factor
        max_freq: Maximum frequency scaling factor. Any value over 1 will
            ignore Nyquists frequency limit

    Returns:
        Sampling frequencies spaced by 1/(T * oversampling) where T
        is the interval spanned by t
    '''

    T = (t.max() - t.min())
    N = t.shape[0]

    return np.arange(1 / (T * oversampling), max_freq * N / (2 * T),
                     1 / (T * oversampling))


def normalize_variance_on(x1: npt.ArrayLike,
                          x2: npt.ArrayLike) -> npt.ArrayLike:
    '''
    Normalize the variance of x1 onto x2
    Args:
        x1: Input time-series
        x2: Reference time-series to match variance of

    Returns:
        x1 with variance scaled to match x2
    '''

    return (np.std(x2) / np.std(x1)) * x1


def ST(t: npt.ArrayLike,
       w: npt.ArrayLike,
       o: npt.ArrayLike,
       a: Optional[int] = 1) -> npt.ArrayLike:
    '''
    Shifted sine function

    Args:
        t: Sampling times
        w: Frequencies
        o: Phase shift
        a: Scaling coefficient

    Return:
        Shifted sine wave
    '''

    return np.sin(a * w * (t - o))


def CT(t: npt.ArrayLike,
       w: npt.ArrayLike,
       o: npt.ArrayLike,
       a: Optional[int] = 1) -> npt.ArrayLike:
    '''
    Shifted cosine function

    Args:
        t: Sampling times
        w: Frequencies
        o: Phase shift
        a: Scaling coefficient

    Return:
        Shifted cos wave
    '''

    return np.cos(a * w * (t - o))


def Ow(t: npt.ArrayLike, w: npt.ArrayLike) -> npt.ArrayLike:
    '''
    Compute frequency dependent coefficient $o$ such that

    $sum_n^N \\cos w(t_n-o) \\sin w(t_n-o) = 0$

    Args:
        t: Time series
        w: Frequencies to find coefficients for

    Returns:
        o: Array of coefficients for each frequency in w
    '''

    return (1 / (2 * w)) * np.arctan2(
        ST(t, w, 0, a=2).sum(axis=0),
        CT(t, w, 0, a=2).sum(axis=0))


def NU_spectral_interpolation(t: npt.ArrayLike, x: npt.ArrayLike,
                    s: npt.ArrayLike,
                    fs: float,
                    chunk_size: Optional[int] = 100) -> npt.ArrayLike:
    '''
    Perform non-uniform spectral interpolation of a time-series t
    to time-series s

    Method is based off of Lomb-Scargle Periodogram

    Args:
        t: Non-uniform sampling times [T x 1]
        x: Observed values x_p(t) [P x T]
        s: Target time-series to sample to [S x 1]
        fs: Sampling frequency
        chunk_size: Perform interpolation in chunks if X is 2 dimensional
            to limit memory consumption

    Returns:
        x_i: Observation array interpolated to S [ P x S ]
    '''

    w = get_sampling_w(t)

    cterm = CT(t, w, 0)
    sterm = ST(t, w, 0)
    o = Ow(t, w)
    X = make_design(s, w)

    coefs = np.empty((CHUNK_SIZE, w.shape[0] * 2))

    dc = (cterm ** 2).sum(axis=0)
    ds = (sterm ** 2).sum(axis=0)

    x_i = np.zeros((s.shape[0], x.shape[0]), dtype=np.float)

    num_chunks = x.shape[1]//(chunk_size + 1)
    chunks = np.array_split(np.arange(0, t.shape[0]), num_chunks)

    for i in chunks:

        coefs = np.empty((i.shape[0], w.shape[0] * 2), dtype=np.float)
        ts = x[i, :]

        nc = x @ cterm
        ns = x @ sterm

        coefs[:, ::2] = nc/dc
        coefs[:, 1::2] = ns/ds

        x_i[i, :] = coefs @ X.T

    return x_i
