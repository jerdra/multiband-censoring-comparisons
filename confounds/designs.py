'''
Common design matrices for temporal filtering operations
'''
from __future__ import annotations
from typing import Optional

import numpy as np
import numpy.typing as npt

from .spectral_interpolation import get_sampling_w


def shifted_fourier(t: npt.ArrayLike,
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


def dct_basis(N: int) -> np.ArrayLike:
    '''
    Compute a DCT basis function for a given time-series
    '''

    n = np.arange(0, N)
    c = np.pi / N
    X = np.empty((N, N))
    for k in n:
        X[:, k] = np.cos(c * (n + 0.5) * k)
    return X


def dct_bandpass(N: int, T: float, low_pass: Optional[float],
                 high_pass: Optional[float]) -> npt.ArrayLike:
    '''
    Compute the cosine basis function for a time-series of length N
    with period T. Will return filtered frequencies with low_pass
    or high_pass options
    '''

    if not low_pass:
        low_pass = 1e15

    if not high_pass:
        high_pass = 0

    n = np.arange(0, N)
    dct = dct_basis(N) * (np.sqrt(2 / N))

    # Remove lowest frequency component
    ft = n / (2 * N * T)[1:]

    pass_band = np.where((ft < low_pass) & (ft > high_pass))[0]
    return dct[:, pass_band]


def fourier_bandpass(N: int, T: float, low_pass: Optional[float],
                     high_pass: Optional[float]) -> npt.ArrayLike:
    '''
    Compute a set of fourier basis functions for a time-series of
    length N with period T. Will returned filtered frequencies with
    low_pass or high_pass options
    '''

    if not low_pass:
        low_pass = 1e15

    if not high_pass:
        high_pass = 0

    t = np.arange(0, N) * T
    w = get_sampling_w(t)

    try:
        passband = np.where((w < low_pass) & (w > high_pass))
    except IndexError:
        # TODO: Add logger error
        raise

    return shifted_fourier(t, w[passband])
