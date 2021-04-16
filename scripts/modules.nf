nextflow.enable.dsl = 2

process scrubImage{

    publishDir "$params.outputdir"

    input:
    tuple val(entities), path(img), path(confounds), val(method)

    output:
    tuple val(entities), val(method),\
    path("${entities}_desc-${method}_cleaned.nii.gz"),\
    emit: clean_img

    shell:
    '''
    python !{workflow.projectDir}/bin/clean_img.py !{img} !{confounds} \
        !{params.cleanconf} !{entities}_desc-!{method}_cleaned.nii.gz \
        --method !{method}
    '''
}

process deriveConnectivity{

    publishDir "$params.outputdir"

    input:
    tuple val(entities), val(method), path(img), path(parcel)

    output:
    tuple val(entities), path("${entities}_desc-${method}_connectivity.npy"),\
    emit: connectivity

    shell:
    '''
    #!/usr/bin/env python

    import numpy as np
    import nilearn.image as nimg
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn.connectome import ConnectivityMeasure

    img = nimg.load_img("!{img}")
    parcel = nimg.load_img("!{parcel}")
    parcel = nimg.resample_to_img(parcel, img, interpolation="nearest")
    smoothing = int("!{params.smoothing_fwhm}")

    masker = NiftiLabelsMasker(labels_img=parcel, smoothing_fwhm=smoothing)
    res = masker.fit_transform(img)

    connecome_measure = ConnectivityMeasure(kind="correlation")
    res = connectome_measure.fit_transform([res])[0]

    np.save("!{entities}_desc-!{method}_connectivity.npy", res)
    '''
}
