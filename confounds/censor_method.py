"""
Provide various strategies to perform cleaning operations
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from nibabel import Nifti1Image
import nilearn.image as nimg
import nilearn.signal as nsig

import logging

from .spectral_interpolation import lombscargle_interpolate
from .designs import dct_bandpass, fourier_bandpass

if TYPE_CHECKING:
    import numpy.typing as npt

logging.basicConfig(format="%(asctime)s %(message)s",
                    datefmt="%m/%d/%Y %I:%M:%S %p",
                    level=logging.INFO)


class BaseClean(object):
    """
    Performs censoring after nilearn.signal.clean step
    """
    def __init__(
        self,
        clean_config: dict,
        low_pass: Optional[float] = None,
        high_pass: Optional[float] = None,
        detrend: Optional[bool] = None,
        standardize: Optional[bool] = None,
        min_contiguous: int = 5,
    ):

        try:
            self._confounds = clean_config["--cf-cols"].split(",")
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
            "standardize": self._standardize,
        }

    def _generate_design(self, confounds: pd.DataFrame) -> npt.ArrayLike:
        return confounds[self._confounds].to_numpy()

    def _get_censor_mask(
            self, fds: npt.ArrayLike,
            fd_thres: float) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Apply Powers et al. 2014 censoring method
        using FD trace.

        Performs initial masking using fd_thres. Then checks
        for blocks with less than a set number of contiguous frames.

        If a block of volumes is less than the number of required contiguous
        frames, it is masked out
        """

        initial_mask = fds.to_numpy() <= fd_thres
        under_min_contiguous = np.zeros_like(initial_mask)
        start_ind = None

        # No frames are censored, then return full set
        if not np.any(np.logical_not(initial_mask)):
            return np.where(initial_mask)[0], np.array([], dtype=np.int)

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

        mask_frames = initial_mask & np.logical_not(under_min_contiguous)
        return (np.where(mask_frames)[0],
                np.where(np.logical_not(mask_frames))[0])

    def _clean(
        self,
        img: Nifti1Image,
        confounds: npt.ArrayLike,
        clean_settings: Optional[dict] = None,
    ) -> Nifti1Image:
        """
        Perform standard Nilearn signals.clean
        """

        if len(img.shape) != 4:
            logging.error("Image is not a time-series image!")
            raise TypeError

        try:
            t_r = img.header["pixdim"][4]
        except IndexError:
            raise

        if not clean_settings:
            clean_settings = self.clean_settings

        return nimg.clean_img(img,
                              confounds=confounds,
                              t_r=t_r,
                              **clean_settings)

    def transform(
        self,
        img: Nifti1Image,
        confounds: pd.DataFrame,
        drop_trs: Optional[int] = None,
        fd_thres: Optional[float] = 0.5,
    ) -> Nifti1Image:

        img, confounds = _clear_steady_state(img, confounds, drop_trs)
        res = self._transform(img, confounds, fd_thres)
        return res

    def _transform(self,
                   img: Nifti1Image,
                   confounds: pd.DataFrame,
                   fd_thres: Optional[float] = 0.5) -> Nifti1Image:
        """
        Perform Naive censoring method on `img`:

        1. Perform cleaning with built-in filtering if specified
        2. Apply censoring with window according to Powers et al. 2014
        and Ciric et al. 2018
        """

        clean_img = self._clean(img, self._generate_design(confounds))
        mask_frames, _ = self._get_censor_mask(
            confounds["framewise_displacement"], fd_thres)

        return _get_vol_index(clean_img, mask_frames)


class PowersClean(BaseClean):
    """
    Implements full Powers 2014 optimized method
    for scrubbing using lombscargle method

    Outline:
    1. Apply censoring to both nuisance regressors and original time-series
    2. Use censored volumes to estimate nuisance regressor coefficients
    3. Apply lomb-scargle interpolation to fill in censored gaps
    4. Filter
    5. Re-censor data
    """
    def _transform(self, img: Nifti1Image, confounds: pd.DataFrame,
                   fd_thres: float) -> Nifti1Image:

        t_r = img.header["pixdim"][4]
        mask_frames, censor_frames = self._get_censor_mask(
            confounds["framewise_displacement"], fd_thres)

        # Step: 1, 2
        confounds = self._generate_design(confounds)[mask_frames, :]
        clean_img = self._clean(_get_vol_index(img, mask_frames), confounds)

        # Step: 3
        clean_img = _interpolate_frames(clean_img, mask_frames, censor_frames,
                                        t_r)

        # Step: 4
        out_img = nimg.clean_img(clean_img,
                                 low_pass=self._low_pass,
                                 high_pass=self._high_pass,
                                 t_r=t_r)

        # Step: 5
        return _get_vol_index(out_img, mask_frames)


class LindquistPowersClean(BaseClean):
    """
    Implements full Powers 2014 optimized method
    modified to satisfy Lindquist's critique

    Outline:
    1. Apply censoring to both nuisance regressors and original time-series
    2. Interpolate censored values using Lombscargle algorithm
    3. Apply filtering to data and nuisance regressors
    4. Re-censor bad volumes
    5. Perform cleaning
    """
    def _censor_and_filter(
        self,
        data: Union[npt.ArrayLike, Nifti1Image],
        mask_frames: npt.ArrayLike,
        censor_frames: npt.ArrayLike,
        t_r: float,
    ) -> Union[npt.ArrayLike, Nifti1Image]:
        """
        Censor data, perform lombscargle interpolation,
        filter
        """

        if isinstance(data, Nifti1Image):
            clean_func = nimg.clean_img
        else:
            clean_func = nsig.clean

        return clean_func(_interpolate_frames(data, mask_frames, censor_frames,
                                              t_r),
                          t_r=t_r,
                          low_pass=self._low_pass,
                          high_pass=self._high_pass,
                          detrend=False,
                          standardize=False)

    def _transform(self, img: Nifti1Image, confounds: pd.DataFrame,
                   fd_thres: float) -> Nifti1Image:

        t_r = img.header["pixdim"][4]
        mask_frames, censor_frames = self._get_censor_mask(
            confounds["framewise_displacement"], fd_thres)

        # Step: 1, 2, 3
        c_img = self._censor_and_filter(img, mask_frames, censor_frames, t_r)
        confounds = self._generate_design(confounds).T
        c_confounds = self._censor_and_filter(confounds, mask_frames,
                                              censor_frames, t_r)

        # Step: 4, 5
        return nimg.clean_img(
            _get_vol_index(c_img, mask_frames),
            confounds=c_confounds[:, mask_frames].T,
            detrend=self._detrend,
            standardize=self._standardize,
        )


class FiltRegressorMixin:
    """
    Specialization of BaseClean to incorporate filtering
    into the regression step in lieu of nilearn's
    temporally dependent butterworth filter.
    """
    def _transform(self,
                   img: Nifti1Image,
                   confounds: pd.DataFrame,
                   fd_thres: Optional[float] = 0.5) -> Nifti1Image:
        """
        Censor data then clean using regression-based filtering
        """

        mask_frames, _ = self._get_censor_mask(
            confounds["framewise_displacement"], fd_thres)
        t_r = img.header["pixdim"][4]

        # Filtering is built into the regressor
        clean_settings = self.clean_settings
        clean_settings.update({"low_pass": None, "high_pass": None})

        clean_img = self._clean(
            _get_vol_index(img, mask_frames),
            self._generate_design(confounds, t_r)[mask_frames, :],
            clean_settings,
        )

        return clean_img


class DCTBasisClean(FiltRegressorMixin, BaseClean):
    """
    Performs simultaneous regression and filtering
    using DCT v1.1.
    """
    def _generate_design(self, confounds: pd.DataFrame,
                         T: float) -> npt.ArrayLike:
        """
        Append DCT regressors to design matrix
        """

        # Generate base design matrix
        X = super()._generate_design(confounds)
        D = dct_bandpass(
            N=confounds.shape[0],
            T=T,
            low_pass=self._low_pass,
            high_pass=self._high_pass,
        )
        return np.c_[X, D]


class FourierBasisClean(FiltRegressorMixin, BaseClean):
    """
    Performs simultaneous regression and filtering
    using Fourier basis
    """
    def _generate_design(self, confounds: pd.DataFrame,
                         T: float) -> npt.ArrayLike:
        """
        Append fourier series basis to design matrix
        """

        X = super()._generate_design(confounds)
        F = fourier_bandpass(
            N=confounds.shape[0],
            T=T,
            low_pass=self._low_pass,
            high_pass=self._high_pass,
        )

        return np.c_[X, F]


def _get_vol_index(img: Nifti1Image, inds: npt.ArrayLike) -> Nifti1Image:
    return nimg.new_img_like(
        img,
        img.get_fdata(caching="unchanged")[:, :, :, inds],
        img.affine,
        copy_header=True,
    )


def _image_to_signals(img: Nifti1Image) -> npt.ArrayLike:
    """
    Transform a Nibabel image into an [NVOX x T] array
    """

    nvox = np.prod(img.shape[:-1])
    return img.get_fdata(caching="unchanged").reshape((nvox, -1))


def _clear_steady_state(
        img: Nifti1Image,
        confounds: pd.DataFrame,
        drop_trs: Optional[int] = None) -> (Nifti1Image, pd.DataFrame):
    """
    Remove steady state volumes

    Arguments:
        img:            Input image
        confounds:      Input confounds dataframe
        drop_trs:       Number of trs to drop

    If no drop_trs are specified and steady_state volumes are not found
    only the first TR will be dropped to allow for derivatives
    """

    if drop_trs is None:
        steady_cols = [c for c in confounds.columns if "steady" in c]
        if steady_cols:
            steady_df = confounds[steady_cols].sum(axis=1).diff()
            steady_ind = np.where(steady_df < 0)[0]
            drop_trs = int(steady_ind[0])
        else:
            logging.warning("No steady state TRs found for this image!")
            drop_trs = 1

    # Construct new image object
    new_conf = confounds.loc[drop_trs:, :]
    new_img = nimg.new_img_like(img,
                                img.get_fdata(caching="unchanged")[:, :, :,
                                                                   drop_trs:],
                                copy_header=True)
    return (new_img, new_conf)


def _interpolate_frames(data: Union[Nifti1Image, npt.ArrayLike],
                        mask: npt.ArrayLike, censor: npt.ArrayLike,
                        t_r: float) -> Union[Nifti1Image, npt.ArrayLike]:
    '''
    Interpolates `censor` using `img` data in `mask`
    using lombscargle non-uniform spectral interpolation

    Arguments:
        data: Input data to censor where last index denotes time-points
            [N1 x ,..., x T]
        mask: Non-censored array indices from uncensored data
        censor: Censored array indices that were removed from `img`
        t_r: Repetition time
    '''

    if not censor.any():
        return data

    is_nifti = False
    if isinstance(data, Nifti1Image):
        sgls = _image_to_signals(data)
        is_nifti = True
    else:
        sgls = data

    t_num_samples = len(censor) + len(mask)

    # Lombscargle interpolate expects already censored data
    if sgls.shape[1] == t_num_samples:
        sgls = sgls[:, mask]

    t = np.arange(0, t_num_samples) * t_r
    interp_vals = lombscargle_interpolate(t=t[mask], x=sgls, s=t)

    res = np.empty((sgls.shape[0], t_num_samples), dtype=sgls.dtype)
    res[:, mask] = sgls
    res[:, censor] = interp_vals[:, censor]
    res = res.reshape((*data.shape[:-1], t_num_samples))

    if is_nifti:
        return nimg.new_img_like(data, res, copy_header=True)

    return res
