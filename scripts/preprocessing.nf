nextflow.enable.dsl = 2

process scrubImage{

    publishDir "$params.outputdir"

    input:
    tuple val(entities), path(img), path(confounds), val(method), \
    val(output)

    output:
    tuple val(entities), val(method),\
    path(output),\
    emit: clean_img

    shell:
    '''
    python !{workflow.projectDir}/bin/clean_img.py !{img} !{confounds} \
        !{params.cleanconf} !{output} \
        --method !{method} \
        !{(params.logDir) ? "--logfile $params.logDir/$entities" + ".log" : ""} 
    '''
}

process deriveConnectivity{

    publishDir "$params.outputdir"

    input:
    tuple val(entities), val(method), path(img), path(parcel),\
    val(output)

    output:
    tuple val(entities), val(method), path(output)

    script:

    def isDtseries = (output.toString().contains("dtseries.nii"))
    def volTableArg = (!isDtseries && params.vol_table) ? "--vol-table ${params.vol_table}":""

    """
    python ${workflow.projectDir}/bin/compute_connectivity.py \
            ${img} ${parcel} ${output} \
            ${(params.logDir) ? "--logfile $params.logDir/$entities" + "_connectivity.log" : ""} \
            ${volTableArg}

    """
}


workflow surfaceCensor{

    take:
        data
        methods
        parcellation

    main:
        i_scrub = data.combine(methods)
                    .map{e,f,c,m -> [
                        e,f,c,m,
                        "${e}_desc-${m}_cleaned.dtseries.nii"
                    ]}
        scrubImage(i_scrub)

        if (parcellation) {
            i_connectivity = scrubImage.out.clean_img
                                .map{e,m,c -> [
                                    e,m,c,
                                    params.surf_parcels,
                                    "${e}_desc-${m}_connectivity.tsv"
                                ]}
            deriveConnectivity(i_connectivity)
        }
}

workflow volumeCensor{

    take:
        data
        methods
        parcellation
        label_table
    
    main:
        i_scrub = data.combine(methods)
            .map{e,f,c,m -> [
                e,f,c,m,
                "${e}_desc-${m}_cleaned.dtseries.nii"
            ]}
        scrubImage(i_scrub)

        if (parcellation) {
            i_connectivity = scrubImage.out.clean_img
                                .map{e,m,c -> [
                                    e,m,c,
                                    params.vol_parcels,
                                    "${e}_desc-${m}_connectivity.tsv"
                                ]}
            deriveConnectivity(i_connectivity)
        }
}
