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
    tuple val(entities), val(method), path(img), path(parcel), \
    path(output)

    output:
    tuple val(entities), path("${entities}_desc-${method}_connectivity.npy"),\
    emit: connectivity

    // Use only if vol_table is available
    script:

    if img.toString().contains("dtseries.nii"){
        let vol_table = "";
    } else {
        let vol_table = params.vol_table ?: ""
    }


    '''
    python !{workflow.projectDir}/bin/compute_connectivity.py \
            !{img} !{parcel} !{output} \
            !{(params.logDir) ? "--logfile $params.logDir/$entities" + "_connectivity.log" : ""} \
            --vol-table !{params.vol_table}

    '''
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
                "${e}_desc-${m}_cleaned.dtseriers.nii"
            ]}
        scrubImage(i_scrub)

}
