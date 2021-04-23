import sys
import argparse
import numpy as np
import logging

import nibabel as nib
from nibabel.cifti2 import Cifti2Image
import pandas as pd


def exception_hook(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught Exception",
                  exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = exception_hook


def volume_connectivity(nifti, parcels):
    '''
    Compute volume-based connectivity and return a dataframe
    containing PxP correlation matrix
    '''
    import nilearn.image as nimg
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn.connectome import ConnectivityMeasure

    parcels = nimg.resample_to_img(parcels, nifti, interpolation='nearest')
    masker = NiftiLabelsMasker(labels_img=parcels)
    res = masker.fit_transform(nifti)

    connectome_measure = ConnectivityMeasure(kind='correlation')
    res = connectome_measure.fit_transform([res])[0]
    return pd.DataFrame(res)


def surface_connectivity(dtseries, dlabel):
    '''
    Compute surface-based connectivity and return a dataframe
    containing PxP correlation matrix
    '''
    ax_bm_ser = dtseries.header.get_axis(1)
    ax_bm_lbl = dlabel.header.get_axis(1)
    ax_lbl = dlabel.header.get_axis(0)

    matched_structs = [(dstruct, dslice, tslice)
                       for dstruct, dslice, _ in ax_bm_lbl.iter_structures()
                       for tstruct, tslice, _ in ax_bm_ser.iter_structures()
                       if dstruct == tstruct]

    ser = dtseries.get_fdata()
    lbl = dlabel.get_fdata().astype(int)

    zts = []
    col_names = []
    for struct, dsl, tsl in matched_structs:
        vtsl = ax_bm_ser.vertex[tsl]

        # Re-order parcellations to monotonically increasing
        # This will help with the array split
        lbl_order = np.argsort(lbl[:, vtsl])[0]
        lbl_slice = lbl[:, vtsl][:, lbl_order]
        ser_slice = ser[:, tsl][:, lbl_order]

        # Group by parcellation, compute meants, standardize
        lbl_ids, lbl_inds = np.unique(lbl_slice, return_index=True)
        arrs = np.split(ser_slice.T, lbl_inds[1:])

        h_zts = np.empty((lbl_ids.shape[0], ser.shape[0]))
        for i, a in enumerate(arrs):
            meants = a.mean(axis=0)
            h_zts[i] = (meants - meants.mean()) / meants.std()

        # Collect result
        zts.append(h_zts)

        # Assuming we have only 1 map
        col_names += [ax_lbl.label[0][i][0] for i in lbl_ids]

    zts = np.vstack(zts)
    R = (zts @ zts.T)
    R /= R[0, 0]
    df = pd.DataFrame(R, columns=col_names)
    df['row'] = col_names
    df = df[['row'] + col_names]
    return df


def configure_logging(logfile):
    '''
    Set up logfile
    '''

    logging.basicConfig(filename=logfile, level=logging.INFO)
    return


def main():

    parser = argparse.ArgumentParser(
        desc="Compute correlation matrix given "
        "an input time-series file and parcellation")

    parser.add_argument('func',
                        type=str,
                        help="Input CIFTI dtseries or "
                        "nifti 4D volume")
    parser.add_argument('parcels',
                        type=str,
                        help="Input parcellation CIFTI dlabel or "
                        "nifti 3D volume with ROI labels")
    parser.add_argument('output', type=str, help="Output CSV file")
    parser.add_argument('--logfile', type=str, help="Output logfile")
    args = parser.parse_args()

    if args.logfile:
        configure_logging(args.logfile)

    img = nib.load(args.func)
    parcels = nib.load(args.parcels)

    if type(img) != type(parcels):
        raise TypeError(f"Type of {args.func} and {args.parcels} "
                        "does not match! Ensure that they are both CIFTI or "
                        "both NIFTI!")

    if isinstance(img, Cifti2Image):
        logging.info("Running surface connectivity...")
        df = surface_connectivity(img, parcels)
    else:
        logging.info("Running volume connectivity...")
        df = volume_connectivity(img, parcels)

    df.to_csv(args.output, delimiter="\t", index=False)
