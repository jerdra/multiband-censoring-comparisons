'''
Perform basic confound cleaning followed by Powers (2014) style scrubbing
'''

import sys
import argparse
import json
import logging

import confounds.censor_method as censor_method
import pandas as pd
import nilearn.image as nimg
import nibabel as nib

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
    p.add_argument("output", help="Output file", type=str)
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

    # Retrieve the censor algorithm
    censor = OBJECT_MAPPING[args.method](config)
    result = censor.transform(nimg.load_img(args.image),
                              pd.read_csv(args.confounds, sep="\t"))
    nib.save(result, args.output)


if __name__ == '__main__':
    main()
