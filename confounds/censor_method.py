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
from .designs import dct_bandpass, fourier_bandpass

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)


class BaseCensor(object):
    '''
    Performs censoring after nilearn.signal.clean step
    '''
    def __init__(self,
                 clean_config: dict,
                 low_pass: Optional[float] = None,
                 high_pass: Optional[float] = None,
                 detrend: Optional[bool] = None,
                 standardize: Optional[bool] = None,
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
        if not self._standardize:
            self._standardize = False

        self._min_contiguous = min_contiguous

    @property
    def clean_settings(self):
        return {
            "low_pass": self._low_pass,
            "high_pass": self._high_pass,
            "detrend": self._detrend,
            "standardize": self._standardize
        }

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

        initial_mask = fds.to_numpy() <= fd_thres
        under_min_contiguous = np.zeros_like(initial_mask)
        start_ind = None

        if not np.any(initial_mask):
            return initial_mask

        for i in np.arange(0, len(initial_mask)):

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

    def _clean(self,
               img: Nifti1Image,
               confounds: pd.DataFrame,
               clean_settings: Optional[dict] = None) -> Nifti1Image:
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

        if not clean_settings:
            clean_settings = self.clean_settings

        return nimg.clean_img(img, t_r=t_r, **clean_settings)

    def transform(self,
                  img: Nifti1Image,
                  confounds: pd.DataFrame,
                  drop_trs: Optional[int] = None,
                  fd_thres: Optional[float] = 0.5) -> Nifti1Image:

        img, confounds = _clear_steady_state(img, confounds, drop_trs)
        res = self._transform(img, confounds, fd_thres)
        return res

    def _transform(self,
                   img: Nifti1Image,
                   confounds: pd.DataFrame,
                   fd_thres: Optional[float] = 0.5) -> Nifti1Image:
        '''
        Perform Naive censoring method on `img`:

        1. Perform cleaning with built-in filtering if specified
        2. Apply censoring with window according to Powers et al. 2014
        and Ciric et al. 2018
        '''

        clean_img = self._clean(img, self._generate_design(confounds))
        censor_frames = self._get_censor_frames(
            confounds['framewise_displacement'], fd_thres)

        censored_img = nimg.new_img_like(
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

    def _transform(self, img: Nifti1Image, confounds: pd.DataFrame,
                   fd_thres: float):

        # Residualize without censored time-series
        censor_frames = self._get_censor_frames(
            confounds['framewise_displacement'], fd_thres)

        clean_img = nimg.clean_img(
            _get_vol_index(img, censor_frames),
            detrend=self._detrend,
            standardize=self._standardize,
            confounds=self._generate_design(confounds)[censor_frames, :])
        clean_img = _image_to_signals(clean_img)

        if not censor_frames.all():
            t_r = img.header['pixdim'][4]
            t = np.arange(0, t_r * img.shape[-1], t_r)[censor_frames]
            s = np.where(np.logical_not(censor_frames))[0] * t_r
            interp_vals = lombscargle_interpolate(t=t,
                                                  x=clean_img,
                                                  s=s,
                                                  fs=1 / t_r)

            non_censor_frames = np.where(np.logical_not(censor_frames))[0]
            clean_img = clean_img.reshape(
                (*img.shape[:-1], censor_frames.sum()))
            interp_vals = interp_vals.reshape(
                (*img.shape[:-1], len(non_censor_frames)))

            out = np.empty(img.shape, dtype=img.get_data_dtype())
            out[:, :, :, np.where(censor_frames)[0]] = clean_img
            out[:, :, :, non_censor_frames] = interp_vals
        else:
            return nimg.new_img_like(img, clean_img, copy_header=True)

        return nimg.new_img_like(img, out, copy_header=True)


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

        # Get data
        t = censor_frames * fs

        # Get values to interpolate on
        interpolate_frames = np.array(
            [i for i in range(len(t)) if i not in censor_frames])
        s = interpolate_frames * fs

        c_data = data[:, censor_frames]

        # Interpolate data indices
        data[:, interpolate_frames] = lombscargle_interpolate(t=t,
                                                              x=c_data,
                                                              s=s,
                                                              fs=fs)
        # Apply filtering on data
        data = nsig.clean(data,
                          low_pass=self._low_pass,
                          high_pass=self._high_pass,
                          t_r=1 / fs,
                          detrend=self._detrend,
                          standardize=self._standardize)
        return data

    def _transform(self, img: Nifti1Image, confounds: pd.DataFrame,
                   fd_thres: float):

        # Get image t_r
        t_r = img.header['pixdim'][4]

        # Apply pre-regression censoring + filtering
        censor_frames = self._get_censor_frames(
            confounds['framewise_displacement'], fd_thres)
        censor_frames = np.where(censor_frames)[0]

        ccf = self._censor_and_filter(
            self._generate_design(confounds).T, censor_frames, 1 / t_r)
        c_data = self._censor_and_filter(_image_to_signals(img), censor_frames,
                                         1 / t_r)

        # Recensor and residualize data
        clean_data = nsig.clean(c_data[:, censor_frames].T,
                                confounds=ccf[:, censor_frames].T,
                                detrend=False,
                                standardize=False)

        # Now copy image and overwrite data
        clean_data = clean_data.T.reshape(
            (*img.shape[:-1], len(censor_frames)))
        return nimg.new_img_like(img, clean_data, copy_header=True)


class FiltRegressorCensor(BaseCensor):
    '''
    Specialization of BaseCensor to incorporate filtering
    into the regression step in lieu of nilearn's
    temporally dependent butterworth filter.
    '''
    def __init__(self,
                 clean_config: dict,
                 low_pass: Optional[float] = None,
                 high_pass: Optional[float] = None,
                 detrend: Optional[bool] = None,
                 standardize: Optional[bool] = None,
                 min_contiguous: int = 5):

        super().__init__(clean_config, low_pass, high_pass, detrend,
                         standardize, min_contiguous)

    def _transform(self,
                   img: Nifti1Image,
                   confounds: pd.DataFrame,
                   fd_thres: Optional[float] = 0.5) -> Nifti1Image:
        '''
        Censor data then clean using regression-based filtering
        '''

        censor_frames = self._get_censor_frames(
            confounds['framewise_displacement'], fd_thres)

        # Filtering is built into the regressor
        clean_settings = self.clean_settings
        clean_settings.update({"low_pass": None, "high_pass": None})

        clean_img = self._clean(_get_vol_index(img, censor_frames),
                                self._generate_design(confounds),
                                clean_settings)

        censored_img = nimg.new_img_like(
            clean_img,
            clean_img.get_fdata(caching="unchanged")[:, :, :, censor_frames],
            clean_img.affine,
            copy_header=True)

        return censored_img


class DCTBasisCensor(FiltRegressorCensor):
    '''
    Performs simultaneous regression and filtering
    using DCT v1.1.
    '''
    def _generate_design(self, confounds: pd.DataFrame,
                         fs: float) -> npt.ArrayLike:
        '''
        Append DCT regressors to design matrix
        '''

        # Generate base design matrix
        X = super()._generate_design(confounds)
        D = dct_bandpass(N=confounds.shape[0],
                         T=1 / fs,
                         low_pass=self._low_pass,
                         high_pass=self._high_pass)
        return np.c_[X, D]


class FourierBasisCensor(FiltRegressorCensor):
    '''
    Performs simultaneous regression and filtering
    using Fourier basis
    '''
    def _generate_design(self, confounds: pd.DataFrame,
                         fs: float) -> npt.ArrayLike:
        '''
        Append fourier series basis to design matrix
        '''

        X = super()._generate_design(confounds)
        F = fourier_bandpass(N=confounds.shape[0],
                             T=1 / fs,
                             low_pass=self._low_pass,
                             high_pass=self._high_pass)

        return np.c_[X, F]


def _get_vol_index(img: Nifti1Image, inds: npt.ArrayLike) -> Nifti1Image:
    return nimg.new_img_like(img,
                             img.get_fdata(caching="unchanged")[:, :, :, inds],
                             img.affine,
                             copy_header=True)


def _image_to_signals(img: Nifti1Image) -> npt.ArrayLike:
    '''
    Transform a Nibabel image into an [NVOX x T] array
    '''

    nvox = np.prod(img.shape[:-1])
    return img.get_fdata(caching='unchanged').reshape((nvox, -1))


def _clear_steady_state(img: Nifti1Image,
                        confounds: pd.DataFrame,
                        drop_trs: Optional[int] = None):
    '''
    Remove steady state volumes
    '''

    if not drop_trs:
        steady_cols = [c for c in confounds.columns if "steady" in c]
        if steady_cols:
            steady_df = confounds[steady_cols].sum(axis=1).diff()
            steady_ind = np.where(steady_df < 0)[0]
            drop_trs = int(steady_ind[0])
        else:
            logging.error("drop_trs not supplied and steady_state volumes not"
                          " found in confounds file!")
            raise ValueError

    # Construct new image object
    new_conf = confounds.loc[drop_trs:, :]
    new_img = nimg.new_img_like(img,
                                img.get_fdata(caching="unchanged")[:, :, :,
                                                                   drop_trs:],
                                copy_header=True)
    return (new_img, new_conf)
