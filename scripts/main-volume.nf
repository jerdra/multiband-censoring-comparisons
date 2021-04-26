nextflow.enable.dsl = 2

include { volumeCensor; surfaceCensor } from "./preprocessing.nf"
include { getMeanFD } from "./utils.nf"

usage = file("${workflow.scriptFile.getParent()}/usage")
printhelp = params.help


req_param = ["--fmriprep": "${params.fmriprep}",
             "--cleanconf": "$params.cleanconf",
             "--outputdir":"$params.outputdir"]

missing_req = req_param.grep{ (it.value == null || it.value == "") }

if (missing_req) {
    log.error("Missing required command-line argument(s)!")
    missing_req.each{ log.error("Missing ${it.key}") }
    printhelp = true
}

if (printhelp) {
    print(usage.text)
    System.exit(0)
}

log.info("Project directory: ${workflow.projectDir}")
log.info("fMRIPrep directory: $params.fmriprep")
if (params.ciftify) {
    log.info("Processing surface data")
    log.info("Ciftify directory: ${params.ciftify}") 
}
log.info("Output directory: $params.outputdir")

// Check if volume table provided with parcellation
if (params.vol_parcels){
    log.info("Volume Parcellation: ${params.vol_parcels}")
    if (!params.vol_table){
        log.info("Volume parcels provided, but not table!")
        log.info("Note that the dataframes outputted will not be labelled" +
        " with ROI names!")
    } else {
        log.info("Volume Table: ${params.vol_table}")
    }
}

if (params.surf_parcels) log.info("Surface Parcellation: ${params.surf_parcels}")


base_entities = /.+(?=_desc)/
// Retrieve all available fMRI files, store with base entities
fmriprep_scans = Channel.empty()
ciftify_scans = Channel.empty()
if (!params.ciftify) {
    fmriprep_scans = Channel.fromPath("$params.fmriprep/sub-*/ses-*/func/" +
                                  "*desc-preproc_bold.nii.gz")
                        .map{p -> [(p.getBaseName() =~ base_entities)[0], p]}
} else {
    ciftify_scans =Channel.fromPath("$params.ciftify/sub-*/MNINonLinear/Results/*" +
                                    "/*dtseries.nii")
                        .map{p -> [(p.getBaseName() =~ base_entities)[0], p]}
                        .map{e, p -> [
                            e, (p.toString() =~ /sub-[A-Za-z0-9]+/)[0], p
                        ]}
                        .map{e, s, p -> [
                            s + "_" + e + "_space-MNI152NLin2009cAsym", p
                        ]}
}

all_scans = fmriprep_scans.mix(ciftify_scans)

// Filter for required bids entities
if (params.bids_entities){
    log.info("Using the following bids entities")
    search_items = params.bids_entities.collect{e -> "${e.key}-${e.value}".toString()}
    search_items.each{ e -> log.info("$e") }
    all_scans = all_scans.filter{ entities, _ ->
        entities.split("_").toList().containsAll(search_items)
    }
}

if (params.subjects){
    log.info("Subjects list provided: $params.subjects")
    subjects_channel = Channel.fromPath(params.subjects)
                        .splitText(){it.strip()}
    all_scans = all_scans.map{e, f ->
        [
            (e =~ /sub-[A-Za-z0-9]+/)[0], e, f
        ]
    }
    .combine(subjects_channel, by: 0)
    .map{s,e,f -> [e,f]}
}

// Join confounds
confounds = Channel.fromPath("$params.fmriprep/sub-*/ses-*/func/"    +
                             "*desc-confounds_timeseries.tsv")
                    .map{p -> [(p.getBaseName() =~ base_entities)[0], p]}

space_entity = ~/_space-[A-Za-z0-9]+/
input_channel = all_scans.map{e, f -> [
    e - space_entity, e, f
]}
.combine(confounds, by: 0)
.map{se, e, f, c -> [e,f,c]}
.branch{
    volumes: it[1].getName().contains("nii.gz")
    surfaces: it[1].getName().contains("dtseries.nii")
}

workflow{

    main:

    surfaceCensor(
        input_channel.surfaces,
        params.METHODS,
        params.surf_parcels
    )

    volumeCensor(
        input_channel.volumes,
        params.METHODS,
        params.vol_parcels,
        params.vol_table
    )

    // Compute meanFD for each entity and save into CSV
    confounds.map{e, c -> "${e},${getMeanFD(c)}"} 
            .collectFile(
                name: "$params.outputdir/meanFD.csv",
                seed: "entity,mean_fd",
                newLine: true
            ).view { "Saved meanFD values into $it" }

}
