'''
Perform basic confound cleaning followed by Powers (2014) style scrubbing
'''

import sys
import argparse
import json
import logging

import numpy as np

import confounds.censor_method as censor_method
import pandas as pd
import nilearn.image as nimg
import nibabel as nib
from nibabel.cifti2 import cifti2_axes, Cifti2Image

OBJECT_MAPPING = {
    "base": censor_method.BaseClean,
    "powers": censor_method.PowersClean,
    "lindquist-powers": censor_method.LindquistPowersClean,
    "dct": censor_method.DCTBasisClean,
    "fourier": censor_method.FourierBasisClean
}


def exception_hook(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught Exception",
                  exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = exception_hook


def simplify_ciftify_cols(config):
    '''
    Squish ciftify sq, sqtd, and td cols into
    single --cf-cols argument
    '''

    new_config = {k: v for k, v in config.items() if "cols" not in k}

    all_cols = config['--cf-cols'].split(',')

    if "--cf-sq-cols" in config.keys():
        all_cols += [f"{f}_power2" for f in config['--cf-sq-cols'].split(',')]

    if "--cf-td-cols" in config.keys():
        all_cols += [
            f"{f}_derivative1" for f in config['--cf-td-cols'].split(',')
        ]

    if "--cf-sqtd-cols" in config.keys():
        all_cols += [
            f"{f}_derivative1_powers2"
            for f in config['--cf-sqtd-cols'].split(',')
        ]

    new_config['--cf-cols'] = ','.join(all_cols)

    return new_config


def convert_to_nifti(cifti):

    # Shape is [T x V]
    data = np.asanyarray(cifti.dataobj).T
    V, T = data.shape

    # We'll convert it into a NIFTI container [V x 1 x 1 x T]
    return nib.Nifti1Image(data[:, np.newaxis, np.newaxis, :], np.eye(4))


def configure_logging(logfile):
    '''
    Set up logging
    '''
    logging.basicConfig(filename=logfile, level=logging.INFO, force=True)
    logging.info('Configured logging')


def main():

    p = argparse.ArgumentParser(
        description="Perform basic confound cleaning followed"
        " by Powers (2014) style scrubbing")
    p.add_argument("image", help="Input image file to be cleaned", type=str)
    p.add_argument("confounds",
                   help="Input confound file to be cleaned",
                   type=str)
    p.add_argument("config",
                   help="Cleaning configuration file"
                   " (see: Ciftify clean config)",
                   type=str)
    p.add_argument("output", help="Output file basename", type=str)
    p.add_argument("--method",
                   help="Censoring method to use",
                   choices=list(OBJECT_MAPPING.keys()))
    p.add_argument("--logfile", help="Output logfile", type=str)

    args = p.parse_args()

    if args.logfile:
        configure_logging(args.logfile)
    with open(args.config, 'r') as f:
        config = json.load(f)
    config = simplify_ciftify_cols(config)

    img = nimg.load_img(args.image)

    # Needs special handling
    surf_header = None
    if isinstance(img, nib.cifti2.Cifti2Image):
        surf_header = img.header
        img = convert_to_nifti(img)

    # Retrieve the censor algorithm
    censor = OBJECT_MAPPING[args.method](config)
    result = censor.transform(img, pd.read_csv(args.confounds, sep="\t"))

    if surf_header is None:
        nib.save(result, args.output)
        return

    # Re-construct surface
    series_ax = surf_header.get_axis(0)
    result_data = np.squeeze(result.get_fdata()).T
    new_header = cifti2_axes.to_header([
        cifti2_axes.SeriesAxis(start=series_ax.start,
                               step=series_ax.step,
                               unit=series_ax.unit,
                               size=result_data.shape[0]),
        surf_header.get_axis(1)
    ])
    cifti = Cifti2Image(dataobj=result_data, header=new_header)
    nib.save(cifti, args.output)


if __name__ == '__main__':
    main()
