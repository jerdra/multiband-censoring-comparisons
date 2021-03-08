'''
Provide various strategies to perform cleaning operations
'''

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nibabel import Nifti1Image
    import numpy.typing as npt

import numpy as np
import pandas as pd
import nilearn.image as nimg
import nilearn.signal as nsig

import logging

from .spectral_interpolation import lombscargle_interpolate

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)


class BaseCensor(object):
    '''
    Performs censoring after nilearn.signal.clean step
    '''
    def __init__(self,
                 clean_config: dict,
                 low_pass: Optional[float],
                 high_pass: Optional[float],
                 detrend: Optional[bool],
                 standardize: Optional[bool],
                 min_contiguous: int = 5):

        try:
            self._confounds = clean_config['--cf-cols'].split(',')
        except IndexError:
            raise

        # Allow for overrides
        self._high_pass = clean_config.get("--high-pass", None) or high_pass
        self._low_pass = clean_config.get("--low-pass", None) or low_pass
        self._detrend = clean_config.get("--detrend", False) or detrend
        self._standardize = clean_config.get("--standardize",
                                             False) or standardize
        self._min_contiguous = min_contiguous

    def _generate_design(self, confounds: pd.DataFrame) -> npt.ArrayLike:
        return confounds[self._confounds].to_numpy()

    def _get_censor_frames(self, fds: npt.ArrayLike, fd_thres: float):
        '''
        Apply Powers et al. 2014 censoring method
        using FD trace.

        Performs initial masking using fd_thres. Then checks
        for blocks with less than a set number of contiguous frames.

        If a block of volumes is less than the number of required contiguous
        frames, it is masked out
        '''

        initial_mask = fds <= fd_thres
        under_min_contiguous = np.zeros_like(initial_mask)
        start_ind = None
        for i in np.arange(0, initial_mask.shape[0]):

            if initial_mask[i] == 1:
                if start_ind:
                    continue
                else:
                    start_ind = i

            if initial_mask[i] == 0 and start_ind:
                if i - start_ind < self._min_contiguous:
                    under_min_contiguous[start_ind:i] = 1
                start_ind = False

        return initial_mask & np.logical_not(under_min_contiguous)

    @property
    def clean_settings(self):
        return {
            "low_pass": self._low_pass,
            "high_pass": self._high_pass,
            "detrend": self._detrend,
            "standardize": self._standardize
        }

    def _clean(self, img: Nifti1Image, confounds: pd.DataFrame) -> Nifti1Image:
        '''
        Perform standard Nilearn signals.clean
        '''

        if len(img.shape) != 4:
            logging.error("Image is not a time-series image!")
            raise TypeError

        try:
            t_r = img.header['pixdim'][4]
        except IndexError:
            raise

        return np.where(nimg.clean_img(img, t_r=t_r, **self.clean_settings))

    def transform(self, img: Nifti1Image, confounds: pd.DataFrame,
                  fd_thres: Optional[float]) -> Nifti1Image:
        '''
        Perform Naive censoring method on `img`:

        1. Perform cleaning with built-in filtering if specified
        2. Apply censoring with window according to Powers et al. 2014
        and Ciric et al. 2018
        '''

        clean_img = self._clean(img, self._generate_design(confounds))
        censor_frames = self._get_censor_frames(confounds['fd'], fd_thres)

        # Using this rather than nilearn.image.index_img
        # since we don't want to store a cached reference
        censored_img = nimg.new_image_like(
            clean_img,
            clean_img.get_fdata(caching="unchanged")[:, :, :, censor_frames],
            clean_img.affine,
            copy_header=True)

        return censored_img


class PowersClean(BaseCensor):
    '''
    Implements full Powers 2014 optimized method
    for scrubbing using lombscargle method

    Outline:
    1. Apply censoring to both nuisance regressors and original time-series
    2. Use censored volumes to estimate nuisance regressor coefficients
    3. Transform original time-series using estimated
       coefficients from clean data
    4. Apply censoring mask again and perform lomb-scargle interpolation
    5. Re-censor data
    '''
    def _clean(self):
        return

    def transform(self, img: Nifti1Image, confounds: pd.DataFrame,
                  fd_thres: float):

        # Residualize without censored time-series
        censor_frames = self._get_censor_frames(confounds["fd"], fd_thres)

        clean_img = nimg.clean(
            _get_vol_index(img, censor_frames),
            detrend=self._detrend,
            standardize=self._standardize,
            confounds=self._generate_design(confounds).loc[censor_frames, :])
        clean_img = _image_to_signals(clean_img)

        t_r = img.header['pixdim'][4]
        t = np.arange(0, t_r * img.shape[1], t_r)[censor_frames]
        s = censor_frames * t_r
        interp_vals = lombscargle_interpolate(t=t, x=clean_img, s=s, fs=t_r)

        non_censor_frames = [
            i for i in range(img.shape[-1]) if i not in censor_frames
        ]
        out = img.get_fdata()
        out = np.empty(img.shape, dtype=img.get_data_dtype())
        out[:, :, :, non_censor_frames] = clean_img.reshape(img.shape)
        out[:, :, :, censor_frames] = interp_vals

        return out


class LindquistPowersClean(BaseCensor):
    '''
    Implements full Powers 2014 optimized method
    modified to satisfy Lindquist's critique

    Outline:
    1. Apply censoring to both nuisance regressors and original time-series
    2. Interpolate censored values using Lombscargle algorithm
    3. Apply filtering to data and nuisance regressors
    4. Re-censor bad volumes
    5. Perform cleaning
    '''
    def _censor_and_filter(self, data: npt.ArrayLike,
                           censor_frames: npt.ArrayLike, fs: float):
        '''
        Censor data, perform lombscargle interpolation,
        filter
        '''

        t = np.arange(0, data.shape[0]) * fs
        s = censor_frames * fs
        c_data = data[:, censor_frames]

        # Interpolate data indices
        data[:, censor_frames] = lombscargle_interpolate(t=t,
                                                         x=c_data,
                                                         s=s,
                                                         fs=fs)
        # Apply filtering on data
        data = nsig.clean(data,
                          low_pass=self._low_pass,
                          high_pass=self._high_pass,
                          t_r=fs,
                          detrend=self._detrend,
                          standardize=self._standardize)
        return data

    def transform(self, img: Nifti1Image, confounds: pd.DataFrame,
                  fd_thres: float):

        # Get image t_r
        t_r = img.header['pixdim'][4]

        # Apply pre-regression censoring + filtering
        censor_frames = self._get_censor_frames(confounds['fd'], fd_thres)
        ccf = self._censor_and_filter(self._generate_design(confounds),
                                      censor_frames, t_r)
        c_data = self._censor_and_filter(img.get_fdata(caching='unchanged'),
                                         censor_frames, t_r)

        # Recensor and residualize data
        clean_data = nsig.clean(c_data[:, censor_frames],
                                confounds=ccf[censor_frames, :],
                                detrend=False,
                                standardize=False)

        # Now copy image and overwrite data
        return nimg.new_img_like(img, clean_data)


class DCTBasisCensor(BaseCensor):
    '''
    Performs simultaneous regression and filtering
    using DCT v1.1
    '''


def _get_vol_index(img: Nifti1Image, inds: npt.ArrayLike) -> Nifti1Image:
    return nimg.new_image_like(img,
                               img.get_fdata(caching="unchanged")[:, :, :,
                                                                  inds],
                               img.affine,
                               copy_header=True)


def _image_to_signals(img: Nifti1Image) -> npt.ArrayLike:
    '''
    Transform a Nibabel image into an [NVOX x T] array
    '''

    nvox = np.prod(img.shape)
    return img.get_fdata(caching='unchanged').reshape((nvox, -1))