nextflow.enable.dsl = 2

process scrubImage{

    publishDir "$params.outputdir"

    input:
    tuple val(entities), path(img), path(confounds)
    each method from params.METHODS

    output:
    tuple val(entities), path("${entities}_desc-${method}_cleaned.nii.gz"),\
    emit: clean_img

    shell:
    '''
    python !{params.bin}/clean_img.py !{img} !{confounds} \
        !{params.cleanconf} !{entities}_desc-!{method}_cleaned.nii.gz \
        --method !{method}
    '''
}
